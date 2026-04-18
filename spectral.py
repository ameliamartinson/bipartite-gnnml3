from graph_loader import *

def phi(x,b=1,f_s=0.5):
    return np.exp(-b*(x - f_s)**2)

def h(x,b=1,f_s=0.5): #even part
    return 0.5*(phi(x,b,f_s) + phi(-x,b,f_s))

def g(x,b=1,f_s=1): #odd part
    return 0.5*(phi(x,b,f_s) - phi(-x,b,f_s))

def spectral_design_full(G: nx.Graph,M: sp.csr_matrix, s,phi,h,g): #
    B = normalized_biadjacency(G).toarray()
    m,n = B.shape

    M = M.toarray()
    M_11 = M[0:m,0:m] #probably all zeros
    M_12 = M[0:m, m:]
    M_21 = M[m:, 0:m]
    M_22 = M[m:, m:] # "

    U,S,Vt = np.linalg.svd(B, full_matrices=False)
    lambda_max = S[0]    

    s_vals = np.linspace(-lambda_max, lambda_max, s)
    print(s_vals)
    #h_S = np.diag(h(S))
    #g_S = np.diag(g(S))

    ret_arr = []
    for f_s in s_vals:
        h_S = np.diag(h(S, f_s=f_s))
        g_S = np.diag(g(S, f_s=f_s))

        UU = M_11*(U@h_S@U.T)
        UV = M_12*(U@g_S@Vt)
        VU = M_21*(Vt.T@g_S@U.T)
        VV = M_22*(Vt.T@h_S@Vt) 

        ret_arr.append(sp.csr_matrix(np.block([[UU,UV],[VU,VV]])))

    return ret_arr

def spectral_design_trunc(G: nx.Graph,M: sp.csr_matrix, s,k,phi,h,g): #
    B = normalized_biadjacency(G)
    m,n = B.shape

    M = M
    M_11 = M[0:m,0:m]
    M_12 = M[0:m, m:]
    M_21 = M[m:, 0:m]
    M_22 = M[m:, m:]

    U,S,Vt = sp.linalg.svds(B, k=6)
    lambda_max = S[0]    

    s_vals = np.linspace(-lambda_max, lambda_max, s)
    print(s_vals)
    #h_S = np.diag(h(S))
    #g_S = np.diag(g(S))

    ret_arr = []
    for f_s in s_vals:
        h_S = np.diag(h(S, f_s=f_s))
        g_S = np.diag(g(S, f_s=f_s))

        UU = M_11*(U@h_S@U.T)
        UV = M_12*(U@g_S@Vt)
        VU = M_21*(Vt.T@g_S@U.T)
        VV = M_22*(Vt.T@h_S@Vt) 

        ret_arr.append(sp.bmat([[UU,UV],[VU,VV]]))

    return ret_arr
