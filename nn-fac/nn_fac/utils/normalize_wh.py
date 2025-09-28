import torch
import warnings
from nn_fac.utils.beta_divergence import gamma_beta

eps = 1e-8 #np.finfo(float).eps

def normalize_WH(W, H, matrix):
    
    if matrix == "H":
        # Normalize so that He = e
        scalH = torch.sum(H, dim=1)
        H = torch.diag(1 / scalH) @ H
        W = W @ torch.diag(scalH)

    elif matrix == "W":
        # Normalize so that W^T e = e
        scalW = torch.sum(W, dim=0)
        H = torch.diag(scalW) @ H
        W = W @ torch.diag(1 / scalW)

    else:
        raise ValueError(f"Matrix must be either 'W' or 'H', but it is {matrix}")

    return W, H

def update_lagragian_multipliers_simplex_projection(C, D, H, beta, lagrangian_multipliers_0, tol = 1e-6, n_iter_max = 100):

    k, n = H.shape
    
    lagrangian_multipliers_0 = lagrangian_multipliers_0.to(torch.float32)
    
    # Comes from 
    Jk1 = torch.ones((k, 1), device=H.device)
    Jn1 = torch.ones(n, device=H.device)

    Jk1 = torch.ones((k,1))
    Jn1 = torch.ones(n)
    ONES = torch.ones((k, n))
    
    lagrangian_multipliers = lagrangian_multipliers_0.clone()


    for iter in range(n_iter_max):

        lagrangian_multipliers_prev = lagrangian_multipliers.clone()
        denom = D - Jk1 @ lagrangian_multipliers.t()
        
        denom = denom + eps  # Avoid division by zero
        #denom = torch.clamp(denom, min=eps) #best option to avoid division by zero than adding eps to D
        #clamp returns a tensor with denom[i] = max(denom[i], eps)
        
        if beta == 1:
            Mat = H * (C / denom)
            Matp = H * (C / (denom**2))
        elif beta == 2:
            Mat = H * (C / denom)**2
            Matp = H * (C / denom)
            Matp = Matp * (C / (denom**2))
        else:
            gb = gamma_beta(beta)
            Mat = H * (C / denom)**gb
            Matp = H * (C / denom)**(gamma_beta(beta-1))
            Matp = Matp * (C / (denom**2))


        xi = torch.sum(Mat, dim=0)
        xi = xi - Jn1
        xip = torch.sum(Matp, dim=0)

        #lagrangian_multipliers = lagrangian_multipliers - (xi / (xip+eps)).reshape((n,1))
        lagrangian_multipliers = lagrangian_multipliers - (xi / (xip + eps)).view(n, 1)

        if torch.max(torch.abs(lagrangian_multipliers - lagrangian_multipliers_prev)) <= tol:
            break

        if iter == n_iter_max - 1:
            warnings.warn('Maximum of iterations reached in the update of the Lagrangian multipliers.')
    return lagrangian_multipliers


#Next, not used for nmf...
"""
# %% Test projection simplex (but doesn't work)
def normalize_W_and_H(W, H, iter):
    #CA MARCHE PAS, C'EST RELOU
    WH = W@H
    simplexed_W = SimplexProjW(W)
    print(f"Avg des simplex: {np.mean(np.amax(simplexed_W, axis = 0))}")
    if np.mean(np.amax(simplexed_W, axis = 0)) > 0.9: # Projection abusive
        columns_norm = W.sum(axis = 0)
        print(columns_norm)
        W = np.maximum(W / columns_norm, eps)
        # Ht = H.T
        # Ht = Ht * columns_norm
        # H = Ht.T
    else:
        W = np.maximum(simplexed_W, eps)
        #H = H / (W.T @ W @ H) #Sur de mon coup là ?

    print(f"Difference max: {np.amax(WH - W@H)}")


    assert (W>0).all()
    assert (np.sum(W, axis = 0) <= 1 + eps*W.shape[0]*W.shape[1]).all()

    return W, H

def SimplexProjW(y):
    ""
    Project y onto the simplex Delta = { x | x >= 0 and sum(x) <= 1 }.
    ""
    x = np.zeros(y.shape)
    for idx in range(y.shape[1]):
        x[:,idx] = ProjectVectorSimplex(y[:,idx])

    return x

def SimplexProjW_valentin(y):
    ""
    Project y onto the simplex Delta = { x | x >= 0 and sum(x) <= 1 }.

    Ne marche pas, je ne sais pas pourquoi (j'ai du mal à comprendre ce bloc)
    ""
    r, m = y.shape
    ys = -np.sort(-y, axis=0)  # Sort in descending order
    lambda_ = np.zeros(m)
    S = np.zeros((r, m))

    for i in range(1, r):
        if i == 1:
            S[i, :] = ys[:i, :] - ys[i, None]
        else:
            S[i, :] = np.sum(ys[:i, :] - ys[i, None], axis=0)

        indi1 = np.where(S[i, :] >= 1)[0]
        indi2 = np.where(S[i, :] < 1)[0]

        if indi1.size > 0:
            if i == 1:
                lambda_[indi1] = -ys[0, indi1] + 1
            else:
                lambda_[indi1] = (1 - S[i - 1, indi1]) / i - ys[i - 1, indi1]

        if i == r - 1:
            lambda_[indi2] = (1 - S[r - 1, indi2]) / r - ys[r - 1, indi2]

    x = np.maximum(y + lambda_, 0)
    return x

def ProjectVectorSimplex(vY):
    # Obtained from https://github.com/RoyiAvital/StackExchangeCodes/blob/master/Mathematics/Q2327504/ProjectSimplexExact.m
    numElements = len(vY)

    if abs(np.sum(vY) - 1) < 1e-9 and np.all(vY >= 0):
        # The input is already within the Simplex.
        vX = vY
        return vX

    vZ = np.sort(vY)

    vParamMu = np.concatenate(([vZ[0] - 1], vZ, [vZ[-1] + 1]))
    hObjFun = lambda paramMu: np.sum(np.maximum(vY - paramMu, 0)) - 1

    vObjVal = np.zeros(numElements + 2)
    for ii in range(numElements + 2):
        vObjVal[ii] = hObjFun(vParamMu[ii])

    if np.any(vObjVal == 0):
        paramMu = vParamMu[vObjVal == 0]
    else:
        # Working on when an Affine Function has the value zero
        valX1Idx = np.where(vObjVal > 0)[0][-1]
        valX2Idx = np.where(vObjVal < 0)[0][0]

        valX1 = vParamMu[valX1Idx]
        valX2 = vParamMu[valX2Idx]
        valY1 = vObjVal[valX1Idx]
        valY2 = vObjVal[valX2Idx]

        paramA = (valY2 - valY1) / (valX2 - valX1)
        paramB = valY1 - (paramA * valX1)
        paramMu = -paramB / paramA

    vX = np.maximum(vY - paramMu, 0)
    return vX
"""

def project_vector_simplex(vY):
    # sort descending
    ys, idx = torch.sort(vY, descending=True)
    cssv = torch.cumsum(ys, dim=0) - 1
    ind = torch.arange(1, vY.numel()+1, device=vY.device)
    cond = ys - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.clamp(vY - theta, min=0)
    return w


def simplex_proj_W(y):
    W = torch.zeros_like(y)
    for j in range(y.shape[1]):
        col = y[:, j]
        w = project_vector_simplex(col)
        W[:, j] = torch.clamp(w, max=1)
    return W