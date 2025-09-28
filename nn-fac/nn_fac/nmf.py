# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:49:25 2019

@author: amarmore
"""

import torch
import time
import math
import warnings

import nn_fac.update_rules.nnls as nnls
import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
import nn_fac.utils.errors as err

from nimfa.methods import seeding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32  

def nmf(data, rank, init = "random", U_0 = None, V_0 = None, n_iter_max=100, tol=1e-8,
        update_rule = "hals", beta = 2,
        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
        verbose=False, return_costs=False, deterministic=False, update_order="UV"):
    """
    ======================================
    Nonnegative Matrix Factorization (NMF)
    ======================================

    Factorization of a matrix M in two nonnegative matrices U and V,
    such that the product UV approximates M.
    If M is of size m*n, U and V are respectively of size m*r and r*n,
    r being the rank of the decomposition (parameter)
    Typically, this method is used as a dimensionality reduction technique,
    or for source separation.

    The objective function is:

        d(M - UV)_{\beta}
        + sparsity_coefficients[0] * (\sum\limits_{j = 0}^{r}||U[:,k]||_1)
        + sparsity_coefficients[1] * (\sum\limits_{j = 0}^{r}||V[k,:]||_1)

    With:

        d(A)_{\beta} the elementwise $\beta$-divergence,
        ||a||_1 = \sum_{i} abs(a_{i}) (Elementwise L1 norm)

    The objective function is minimized by fixing alternatively
    one of both factors U and V and optimizing on the other one.
    More precisely, the chosen optimization algorithm is the HALS [1],
    which updates each factor columnwise, fixing every other columns,
    each subproblem being reduced to a Nonnegative Least Squares problem 
    if the update_rule is "hals",
    or by using the Multiplicative Update [4,5] on each factor
    if the update_rule is "mu".
    The MU is minimizes the $\beta$-divergence,
    whereas the HALS minimizes the Frobenius norm only.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized
    rank: integer
        The rank of the decomposition
    init: "random" | "nndsvd" | "custom" |
        - If set to random:
            Initialize with random factors of the correct size.
            The randomization is the uniform distribution in [0,1),
            which is the default from numpy random.
        - If set to nnsvd:
            Corresponds to a Nonnegative Double Singular Value Decomposition
            (NNDSVD) initialization, which is a data based initialization,
            designed for NMF. See [2] for details.
            This NNDSVD if performed via the nimfa toolbox [3].
        - If set to custom:
            U_0 and V_0 (see below) will be used for the initialization
        Default: random
    U_0: None or array of nonnegative floats
        A custom initialization of U, used only in "custom" init mode.
        Default: None
    V_0: None or array of nonnegative floats
        A custom initialization of V, used only in "custom" init mode.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in cost function value.
        Between two succesive iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "mu"
        The chosen update rule.
        HALS performs optimization with the euclidean norm,
        MU performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
        Default: "hals"
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
        Default: 2
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
    verbose: boolean
        Indicates whether the algorithm prints the successive
        normalized cost function values or not
        Default: False
    return_costs: boolean
        Indicates whether the algorithm should return all normalized cost function 
        values and computation time of each iteration or not
        Default: False
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).
        Default: False

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    cost_fct_vals: list
        A list of the normalized cost function values, for every iteration of the algorithm.
    toc: list
        A list with accumulated time for every iterations

    Example
    -------
    >>> import numpy as np
    >>> from nn_fac import nmf
    >>> rank = 5
    >>> U_lines = 100
    >>> V_col = 125
    >>> U_0 = np.random.rand(U_lines, rank)
    >>> V_0 = np.random.rand(rank, V_col)
    >>> M = U_0@V_0
    >>> U, V = nmf.nmf(M, rank, init = "random", n_iter_max = 500, tol = 1e-8,
               sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
               verbose=True, return_costs = False)

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2]: C. Boutsidis and E. Gallopoulos. "SVD based
    initialization: A head start for nonnegative matrix factorization,"
    Pattern Recognition 41.4 (2008), pp. 1350{1362.

    [3]: B. Zupan et al. "Nimfa: A python library for nonnegative matrix
    factorization", Journal of Machine Learning Research 13.Mar (2012),
    pp. 849{853.
    
    [4] Févotte, C., & Idier, J. (2011). 
    Algorithms for nonnegative matrix factorization with the β-divergence. 
    Neural computation, 23(9), 2421-2456.
    
    [5] Lee, D. D., & Seung, H. S. (1999). 
    Learning the parts of objects by non-negative matrix factorization.
    Nature, 401(6755), 788-791.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, device=device, dtype=dtype)

    if min(data.shape) < rank:
        min_data = min(data.shape)
        rank = min_data
        warnings.warn(f"The rank is too high for the input matrix. It was set to {min_data} instead.")

    if init.lower() == "random":
        k, n = data.shape
        gen = torch.Generator(device=device)
        if deterministic:
            gen.manual_seed(82)
            U_0 = torch.rand(k, rank, device=device, generator=gen)
            V_0 = torch.rand(rank, n, device=device, generator=gen)
        else:
            U_0 = torch.rand(k, rank, device=device)
            V_0 = torch.rand(rank, n, device=device)

    elif init.lower() == "nndsvd":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # A warning arises from the nimfa toolbox, because of the sue of np.asmatrix.
            #U_0, V_0 = seeding.Nndsvd().initialize(data, rank, {'flag': 0})
            U0_np, V0_np = seeding.Nndsvd().initialize(data.cpu().numpy(), rank, {'flag': 0})

        U_0 = torch.tensor(U0_np + 1e-12, device=device)
        V_0 = torch.tensor(V0_np + 1e-12, device=device)

    elif init.lower() == "custom":
        if U_0 is None or V_0 is None:
            raise Exception("Custom initialization, but one factor is set to 'None'")
        U_0 = torch.tensor(U_0, device=device, dtype=dtype)
        V_0 = torch.tensor(V_0, device=device, dtype=dtype)

    else:
        raise Exception('Initialization type not understood')

    return compute_nmf(data, rank, U_0, V_0, n_iter_max=n_iter_max, tol=tol,
                       update_rule = update_rule, beta = beta,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                       verbose=verbose, return_costs=return_costs, deterministic=deterministic, update_order=update_order)

# Author : Jeremy Cohen, modified by Axel Marmoret
def compute_nmf(data, rank, U_in, V_in, n_iter_max=100, tol=1e-8,
                update_rule = "hals", beta = 2,
                sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
                verbose=False, return_costs=False, deterministic=False, update_order="UV"):
    """
    Computation of a Nonnegative matrix factorization via
    hierarchical alternating least squares (HALS) [1],
    or Multiplicative Update (MU) [2],
    with U_in and V_in as initialization.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    U_in: array of floats
        Initial U factor, of size m*r
    V_in: array of floats
        Initial V factor, of size r*n
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in cost function value.
        Between two iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "mu"
        The chosen update rule.
        HALS performs optimization with the euclidean norm,
        MU performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
        Default: "hals"
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
        Default: 2
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
    verbose: boolean
        Indicates whether the algorithm prints the successive
        normalized cost function values or not
        Default: False
    return_costs: boolean
        Indicates whether the algorithm should return all normalized cost function 
        values and computation time of each iteration or not
        Default: False
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).
        Default: False

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    cost_fct_vals: list
        A list of the normalized cost function values, for every iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] Févotte, C., & Idier, J. (2011). 
    Algorithms for nonnegative matrix factorization with the β-divergence. 
    Neural computation, 23(9), 2421-2456.
    """
    # initialisation
    #U = U_in.copy()
    #V = V_in.copy()
    U = U_in.detach().clone()#
    V = V_in.detach().clone()#
    
    cost_fct_vals = []
    norm_data = torch.linalg.norm(data)
    tic = time.time()
    toc = []

    if sparsity_coefficients == None:
        sparsity_coefficients = [None, None]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or normalize == False:
        normalize = [False, False]

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        U, V, cost = one_nmf_step(data, rank, U, V, norm_data, update_rule, beta,
                                  sparsity_coefficients, fixed_modes, normalize, deterministic, update_order)

        toc.append(time.time() - tic)

        cost_fct_vals.append(cost)

        if verbose:
            if iteration == 0:
                print('Normalized cost function value={}'.format(cost))
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    print('Normalized cost function value={}, variation={}.'.format(
                            cost_fct_vals[-1], cost_fct_vals[-2] - cost_fct_vals[-1]))
                else:
                    # print in red when the reconstruction error is negative (shouldn't happen)
                    print('\033[91m' + 'Normalized cost function value={}, variation={}.'.format(
                            cost_fct_vals[-1], cost_fct_vals[-2] - cost_fct_vals[-1]) + '\033[0m')

        if iteration > 0 and abs(cost_fct_vals[-2] - cost_fct_vals[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print('Converged in {} iterations.'.format(iteration))
            break

    if return_costs:
        return U, V, cost_fct_vals, toc
    return U, V

def one_nmf_step(data, rank, U_in, V_in, norm_data, update_rule, beta,
                 sparsity_coefficients, fixed_modes, normalize, deterministic, update_order="UV"):
    """
    One pass of updates for each factor in NMF
    Update the factors by solving a nonnegative least squares problem per mode
    if the update_rule is "hals",
    or by using the Multiplicative Update on each factor
    if the update_rule is "mu".

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    U_in: array of floats
        Initial U factor, of size m*r
    V_in: array of floats
        Initial V factor, of size r*n
    norm_data: float
        The Frobenius norm of the input matrix (data)
    update_rule: string "hals" | "mu"
        The chosen update rule.
        HALS performs optimization with the euclidean norm,
        MU performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
    normalize: List of boolean (two)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).
    update_order: string "UV" | "VU"
        The order of update of the factors.
        "UV" means that U is updated first, then V.
        "VU" means that V is updated first, then U.

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    cost_fct_val:
        The value of the cost function at this step,
        normalized by the squared norm of the original matrix.
    """
    if update_rule not in ["hals", "mu"]:
        raise err.InvalidArgumentValue(f"Invalid update rule: {update_rule}") from None
    if update_rule == "hals" and beta != 2:
        raise err.InvalidArgumentValue(f"The hals is only valid for the frobenius norm, corresponding to the beta divergence with beta = 2. Here, beta was set to {beta}. To compute NMF with this value of beta, please use the mu update_rule.") from None

    if len(sparsity_coefficients) != 2:
        raise ValueError("NMF needs 2 sparsity coefficients to be performed")

    # Copy
    #U = U_in.copy()
    #V = V_in.copy()

    #print(f"type of U_in: {type(U_in)}, type of V_in: {type(V_in)}, data type: {type(data)}")
    U = U_in.detach().clone().to(device=device, dtype=dtype)   #torch.tensor(U_in, device=device, dtype=dtype)
    V = V_in.detach().clone().to(device=device, dtype=dtype)    #torch.tensor(V_in, device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)
    #print(f"type of U_in: {type(U_in)}, type of V_in: {type(V_in)}, data type: {type(data)}")
    

    def update_U(): #update U
        nonlocal U
        if 0 not in fixed_modes:
            # U update
            
            if update_rule == "hals":
                # Set timer for acceleration in hals_nnls_acc
                tic = time.time()
        
                # Computing cross products
                VVt = V @ V.t()
                VMt = V @ data.t()
        
                # End timer for acceleration in hals_nnls_acc
                timer = time.time() - tic

                with torch.no_grad():    
                    # Compute HALS/NNLS resolution
                    """
                    if deterministic:
                        U = np.transpose(nnls.hals_nnls_acc(VMt, VVt, np.transpose(U_in), maxiter=100, atime=timer, alpha=math.inf, delta=0.01,
                                                        sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False)[0])
                    else:
                        U = np.transpose(nnls.hals_nnls_acc(VMt, VVt, np.transpose(U_in), maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                                                        sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False)[0])
                    """
                
                    VMt_np = VMt.cpu().numpy()
                    VVt_np = VVt.cpu().numpy()

                    alpha_val = math.inf if deterministic else 0.5
                    U_new_np = nnls.hals_nnls_acc(VMt_np, VVt_np, U_in.t().cpu().numpy(),
                                                maxiter=100, atime=timer, delta=0.01,                                        
                                                alpha=alpha_val, sparsity_coefficient = sparsity_coefficients[0], 
                                                normalize = normalize[0], nonzero = False)[0]
                    U = torch.from_numpy(U_new_np.T).to(device=device, dtype=dtype) #'cause nnls returns np.array
                
            elif update_rule == "mu":
                U = mu.switch_alternate_mu(data, U, V, beta, "U") #mu.mu_betadivmin(U, V, data, beta)

    def update_V(): #update V
        nonlocal V
        if 1 not in fixed_modes:
            # V update

            if update_rule == "hals":
                # Set timer for acceleration in hals_nnls_acc
                tic = time.time()
        
                # Computing cross products
                #print(f"type of U: {type(U)}, type of data: {type(data)}")
                #print(f"U shape: {U.shape}, data shape: {data.shape}")
                UtU = U.t() @ U
                UtM = U.t() @ data
        
                # End timer for acceleration in hals_nnls_acc
                timer = time.time() - tic
        
                with torch.no_grad():
                    # Compute HALS/NNLS resolution
                    """
                    if deterministic:
                        V = nnls.hals_nnls_acc(UtM, UtU, V_in, maxiter=100, atime=timer, alpha=math.inf, delta=0.01,
                                        sparsity_coefficient = sparsity_coefficients[1], normalize = normalize[1], nonzero = False)[0]
                    else:
                        V = nnls.hals_nnls_acc(UtM, UtU, V_in, maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                                        sparsity_coefficient = sparsity_coefficients[1], normalize = normalize[1], nonzero = False)[0]
                    """
                    UtU_np = UtU.cpu().numpy()
                    UtM_np = UtM.cpu().numpy()
                    V_in_np = V_in.cpu().numpy()

                    alpha_val = math.inf if deterministic else 0.5
                    V_new_np = nnls.hals_nnls_acc(UtM_np, UtU_np, V_in_np,
                                                maxiter=100, atime=timer, delta=0.01,
                                                alpha=alpha_val, sparsity_coefficient = sparsity_coefficients[1],
                                                normalize = normalize[1], nonzero = False)[0]
                    V = torch.from_numpy(V_new_np).to(device=device, dtype=dtype)

            elif update_rule == "mu":
                V = mu.switch_alternate_mu(data, U, V, beta, "V") # np.transpose(mu.mu_betadivmin(V.T, U.T, data.T, beta))

    if update_order.upper() == "UV":
        update_U()
        update_V()
    elif update_order.upper() == "VU":
        update_V()
        update_U()
    else:
        raise err.InvalidArgumentValue(f"Invalid update order: {update_order}. Must be 'UV' or 'VU'") from None
    
    #sparsity_coefficients = np.where(np.array(sparsity_coefficients) == None, 0, sparsity_coefficients)
    sparsity_coefficients = torch.tensor([0.0 if s is None else s for s in sparsity_coefficients],
                                        dtype=dtype, device=device)

    if update_rule == "hals":
        #cost = np.linalg.norm(data-np.dot(U,V), ord='fro') ** 2 + 2 * (sparsity_coefficients[0] * np.linalg.norm(U, ord=1) + sparsity_coefficients[1] * np.linalg.norm(V, ord=1))
        recon = U @ V
        fro = torch.norm(data - recon)**2
        l1 = 2*( (sparsity_coefficients[0])*torch.norm(U,p=1) + #or sparsity_coefficients[0] or 0
                 (sparsity_coefficients[1])*torch.norm(V,p=1) )
        cost = fro + l1

    elif update_rule == "mu":
        cost = beta_div.beta_divergence(data, U @ V, beta)

    #cost = cost/(norm_data**2)
    return U, V, cost

if __name__ == "__main__":
    """
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    W_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, n) # Example input matrices
    data = W_0@H_0 + 1e-2*np.random.rand(m,n)  # Example input matrix
    
    W, H = nmf(data, rank, beta = 2, update_rule = "hals", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 2, update_rule = "hals", n_iter_max = 100, init = "nndsvd",verbose = True)

    W, H = nmf(data, rank, beta = 1, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 1, update_rule = "mu", n_iter_max = 100, init = "nndsvd",verbose = True)

    W, H = nmf(data, rank, beta = 0, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 0, update_rule = "mu", n_iter_max = 100, init = "nndsvd",verbose = True)

    """
    # Example
    m,n,r = 100,200,5

    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.randn(m,n, device=device).abs()
    data = data.numpy()
    
    U,V = nmf(data, r, init='nndsvd', n_iter_max=10, tol=1e-8, update_rule="mu",
              beta=0, verbose=True)
    
    print("U shape:", U.shape)
    print("V shape:", V.shape)
    
    print(data[0, :5])
    