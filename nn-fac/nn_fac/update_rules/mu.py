# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: amarmore

## Author : Axel Marmoret, based on Florian Voorwinden's code during its internship.

Upgraded by Mahamat Mahamat Nour Bachar(@mahamat9) to use PyTorch tensors and GPU acceleration.

"""

import torch
import tensorly as tl #not used in this version with PyTorch tensors

import nn_fac.utils.errors as err
from nn_fac.utils.beta_divergence import gamma_beta
import nn_fac.utils.normalize_wh as normalize_wh

epsilon = 1e-12

dtype = torch.float32  # ou torch.float64 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def switch_alternate_mu(data, U, V, beta, matrix):
    """
    Encapsulates the switch between the two multiplicative update rules.
    """

    U = U.to(device=device, dtype=dtype)
    V = V.to(device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)

    if matrix in ["U", "W"]:
        return mu_betadivmin(U, V, data, beta)
    elif matrix in ["V", "H"]:
        return mu_betadivmin(V.t(), U.t(), data.t(), beta).t()
    else:
        raise err.InvalidArgumentValue(f"Invalid value for matrix: got {matrix}, but it must be 'U' or 'W' for the first matrix, and 'V' or 'H' for the second one.") from None

def mu_betadivmin(U, V, M, beta):
    """
    =====================================================
    Beta-Divergence NMF solved with Multiplicative Update
    =====================================================

    Computes an approximate solution of a beta-NMF
    [3] with the Multiplicative Update rule [2,3].
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    Conversely than in [1], the NNLS problem is solved for the beta-divergence,
    as studied in [3]:

            min_{U >= 0} beta_div(M, UV)

    The update rule of this algorithm is defined in [3].

    Parameters
    ----------
    U : m-by-r array
        The first factor of the NNLS, the one which will be updated.
    V : r-by-n array
        The second factor of the NNLS, which won't be updated.
    M : m-by-n array
        The initial matrix, to approach.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    U: array
        a m-by-r nonnegative matrix \approx argmin_{U >= 0} beta_div(M, UV)

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.

    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    K = U @ V #torch.matmul(U,V)

    if beta == 1:
        K_inverted = torch.pow(K + epsilon, -1)    
        numer = (K_inverted * M) @ V.t()
        line = torch.sum(V, dim=1, keepdim=True).t()
        denom = line.repeat(K.shape[0], 1)
        U_new = U * numer / (denom + epsilon)
    elif beta == 2:
        numer = M @ V.t()
        denom = (K @ V.t())
        U_new = U * numer / (denom + epsilon)
    elif beta == 3:
        gb = gamma_beta(beta)
        numer = ((K * M) @ V.t()) ** gb
        denom = (K**2 @ V.t()) ** gb
        U_new = U * numer / denom
    else:
        gb = gamma_beta(beta)
        numer = ((K**(beta-2) * M) @ V.t()) ** gb
        denom = (K**(beta-1) @ V.t()) ** gb
        U_new = U * numer / denom
    return torch.clamp(U_new, min=epsilon)


def mu_tensorial(G, factors, tensor, beta):
    """
    This function is used to update the core G of a
    nonnegative Tucker Decomposition (NTD) [1] with beta-divergence [3]
    and Multiplicative Updates [2].

    See ntd.py of this module for more details on the NTD (or [1])

    TODO: expand this docstring.

    Parameters
    ----------
    G : tensorly tensor
        Core tensor at this iteration.
    factors : list of tensorly tensors
        Factors for NTD at this iteration.
    T : tensorly tensor
        The tensor to estimate with NTD.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    G : tensorly tensor
        Update core in NTD.

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.

    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    #G = G.to(torch.float32)
    factors = [f for f in factors] #[f.to(torch.float32) for f in factors]
    #tensor = tensor.to(torch.float32)
    
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    K = tl.tenalg.multi_mode_dot(G,factors)

    if beta == 1:
        L1 = torch.ones_like(K)
        L2 = K.pow(-1) * tensor
        
    elif beta == 2:
        L1 = K
        L2 = torch.ones_like(K) * tensor

    elif beta == 3:
        L1 = K.pow(2)
        L2 = K * tensor

    else:
        L1 = K.pow(beta-1)
        L2 = K.pow(beta-2) * tensor


    numerator = tl.tenalg.multi_mode_dot(L2, [fac.t() for fac in factors])
    denominator = tl.tenalg.multi_mode_dot(L1, [fac.t() for fac in factors])
    G_new = G * (numerator / denominator) ** gamma_beta(beta)

    return torch.clamp(G_new, min=epsilon)

def simplex_proj_mu(data, W, H, beta, tol_update_lagrangian = 1e-6):
    # Projects H on the unit simplex, comes from 'Leplat, V., Gillis, N., & Idier, J. (2021). Multiplicative updates for NMF with β-divergences under disjoint equality constraints. SIAM Journal on Matrix Analysis and Applications, 42(2), 730-752. arXiv:2010.16223.'

    #W = W.to(torch.float32)
    #H = H.to(torch.float32)
    #data = data.to(torch.float32)
    
    k, n = H.shape
    Jk1 = torch.ones((k, 1), device=H.device)
    C = W.t() @ ((W @ H).pow(beta-2) * data)
    D = W.t() @ (W @ H).pow(beta-1)


    lagrangian_multipliers_0 = (D[0, :] - C[0, :] * H[0, :]).pow(gamma_beta(beta)).t()
    lagrangian_multipliers_0 = lagrangian_multipliers_0.view(n, 1)#or .reshape((n, 1))
    lagrangian_multipliers = normalize_wh.update_lagragian_multipliers_simplex_projection(C, D, H, beta, lagrangian_multipliers_0, tol = tol_update_lagrangian, n_iter_max = 100)

    H = H * (C / ((D - Jk1 @ lagrangian_multipliers.t()) + epsilon)).pow(gamma_beta(beta))
    H = torch.clamp(H, min=epsilon)
    
    return H
