"""
Run the reference LightGCN (LightGCN-PyTorch) under the SAME protocol as our
GNNML3 harness and emit a result record in the shared schema (bench_utils).

It reuses LightGCN's own model, sampler, BPR loss, and Test() metric code
unchanged, so the only thing this wrapper adds is: matched flags, seeding,
best-epoch selection, efficiency instrumentation, and writing one JSONL row.

Example:
    python run_lightgcn.py --dataset gowalla --seed 2020 --epochs 1000 --topks "[20]"
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_LGN_CODE = os.path.join(_HERE, "LightGCN-PyTorch", "code")
_LGN_DATA = os.path.join(_HERE, "LightGCN-PyTorch", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="gowalla", choices=["amazon-book", "gowalla", "yelp2018"]
    )
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--recdim", type=int, default=64, help="embedding dim")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--topks", default="[20]")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "benchmark.jsonl"))
    ap.add_argument(
        "--save-model",
        default="",
        help="path to save a checkpoint of the best model (state_dict + final "
        "user/item embeddings, consumable by recommend.py); empty = don't save",
    )
    args = ap.parse_args()

    # LightGCN's `world` module parses sys.argv at import time, so construct the
    # argv it expects BEFORE importing it. We point it at our matched flags and
    # disable tensorboard.
    sys.argv = [
        "lightgcn",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--recdim", str(args.recdim),
        "--layer", str(args.layer),
        "--lr", str(args.lr),
        "--decay", str(args.decay),
        "--topks", args.topks,
        "--tensorboard", "0",
    ]

    # Import LightGCN's modules from its code dir; build the dataset with an
    # absolute path so this works regardless of CWD.
    sys.path.insert(0, _LGN_CODE)
    sys.path.insert(0, _HERE)
    import torch
    import world  # parses sys.argv above
    import utils
    import dataloader
    import model as lgn_model
    import Procedure
    from bench_utils import append_jsonl

    if args.device != "auto":
        world.device = torch.device(args.device)
    device = world.device
    print(f"LightGCN device: {device}")

    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)

    ks = list(eval(args.topks))
    primary_k = ks[0]
    # world.topks drives Procedure.Test's output ordering.
    world.topks = ks

    t_setup = time.time()
    dataset = dataloader.Loader(
        config=world.config, path=os.path.join(_LGN_DATA, args.dataset)
    )
    Recmodel = lgn_model.LightGCN(world.config, dataset).to(device)
    setup_time_s = time.time() - t_setup

    bpr = utils.BPRLoss(Recmodel, world.config)
    n_params = sum(p.numel() for p in Recmodel.parameters())
    print(f"  Params: {n_params:,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_test(epoch):
        res = Procedure.Test(dataset, Recmodel, epoch, None, world.config["multicore"])
        # res['recall'/'ndcg'/'precision'] are arrays aligned to world.topks (= ks)
        return {
            **{f"recall@{k}": float(res["recall"][i]) for i, k in enumerate(ks)},
            **{f"ndcg@{k}": float(res["ndcg"][i]) for i, k in enumerate(ks)},
            **{f"precision@{k}": float(res["precision"][i]) for i, k in enumerate(ks)},
        }

    def save_checkpoint(rec, epoch):
        """Snapshot the state dict plus the propagated (post-LGC) user/item
        embeddings, in the same schema bipartite_experiment.py saves, so
        recommend.py can consume either model's checkpoint."""
        Recmodel.eval()
        with torch.no_grad():
            ue, ie = Recmodel.computer()
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
        torch.save(
            {
                "model": "lightgcn",
                "dataset": args.dataset,
                "epoch": epoch,
                "metrics": {m: float(v) for m, v in rec.items()},
                "config": vars(args),
                "state_dict": Recmodel.state_dict(),
                "user_emb": ue.cpu(),
                "item_emb": ie.cpu(),
            },
            args.save_model,
        )

    best = None
    best_epoch = 0
    time_to_best_s = 0.0
    train_start = time.time()

    for epoch in range(args.epochs):
        if epoch % args.eval_every == 0:
            rec = run_test(epoch)
            if best is None or rec[f"recall@{primary_k}"] > best[f"recall@{primary_k}"]:
                best, best_epoch = rec, epoch
                time_to_best_s = time.time() - train_start
                if args.save_model:
                    save_checkpoint(rec, epoch)
        Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=None)

    # Final evaluation after the last training step.
    rec = run_test(args.epochs)
    if best is None or rec[f"recall@{primary_k}"] > best[f"recall@{primary_k}"]:
        best, best_epoch = rec, args.epochs
        time_to_best_s = time.time() - train_start
        if args.save_model:
            save_checkpoint(rec, args.epochs)

    train_time_s = time.time() - train_start
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )

    print(f"\nBest epoch {best_epoch} (selected by recall@{primary_k}):")
    for k in ks:
        print(
            f"  Recall@{k}: {best[f'recall@{k}']:.4f}  "
            f"NDCG@{k}: {best[f'ndcg@{k}']:.4f}  "
            f"Precision@{k}: {best[f'precision@{k}']:.4f}"
        )

    row = {
        "model": "lightgcn",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "embed_dim": args.recdim,
        "layers": args.layer,
        "lr": args.lr,
        "decay": args.decay,
        "device": str(device),
        "best_epoch": best_epoch,
        "setup_time_s": round(setup_time_s, 3),
        "train_time_s": round(train_time_s, 3),
        "time_to_best_s": round(time_to_best_s, 3),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "n_params": int(n_params),
    }
    for k in ks:
        row[f"recall@{k}"] = round(best[f"recall@{k}"], 6)
        row[f"ndcg@{k}"] = round(best[f"ndcg@{k}"], 6)
        row[f"precision@{k}"] = round(best[f"precision@{k}"], 6)

    append_jsonl(args.out, row)
    print(f"\nResult appended to {args.out}")
    if args.save_model:
        print(f"Best-epoch checkpoint saved to {args.save_model}")


if __name__ == "__main__":
    main()
