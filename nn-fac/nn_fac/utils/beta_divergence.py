# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: amarmore    

## Author : Axel Marmoret, based on Florian Voorwinden's code during its internship.

Upgraded by Mahamat Mahamat Nour Bachar(@mahamat9) to use PyTorch tensors and GPU acceleration.

"""

import torch
import nn_fac.utils.errors as err


def beta_divergence(a, b, beta):
    """
    Compute the beta-divergence of two tensors a and b using PyTorch.

    Parameters
    ----------
    a : float or array
        First argument for the beta-divergence.
    b : float or array
        Second argument for the beta-divergence. 
    beta : float
        the beta factor of the beta-divergence.
    
    Returns
    -------
    float
        Beta-divergence of a and b.
        
    References
    ----------
    [1] C. Févotte and J. Idier, Algorithms for nonnegative matrix 
    factorization with the beta-divergence, Neural Computation, 
    vol. 23, no. 9, pp. 2421–2456, 2011.
    
    """
    # Ensure tensors
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative.")

    if beta == 1:
        # KL divergence: sum(a * log(a/b) - a + b)
        # add small eps to avoid log(0)
        eps = 1e-12
        ratio = a / (b + eps)
        return torch.sum(a * torch.log(ratio + eps) - a + b)
    elif beta == 0:
        # IS divergence: sum(a/b - log(a/b) - 1)
        eps = 1e-12
        ratio = a / (b + eps)
        return torch.sum(ratio - torch.log(ratio + eps) - 1)
    else:
        # general case
        return torch.sum((a**beta + (beta - 1) * b**beta - beta * a * b**(beta - 1)) / (beta * (beta - 1)))


def gamma_beta(beta):
    """
    Exponent of Fevotte and Idier [1], which guarantees the MU updates decrease the cost.
    
    See [1] for details.
    
    Parameters
    ----------
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    int : the exponent value
    
    References
    ----------
    [1]  C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """
    if beta < 1:
        return 1.0 / (2 - beta)
    if beta > 2:
        return 1.0 / (beta - 1)
    return 1.0
