from cil.optimisation.functions import Function

import numpy as np
from numba import jit, prange


class Fair(Function):
    
    """                 
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, theta, **kwargs):

        super(Fair, self).__init__(L=1/theta)  
        self.theta = theta

    def __call__(self, x , out = None):

        res = x.clone()

        norm = x.pnorm(2)
        norm_arr = norm.as_array().flatten()

        for el in res:
            el_arr = el.as_array()
            _get_value_numba(el_arr, norm_arr, self.theta)

        if out is None:
            return res.sum()

        else:
            out = res.sum()

    def gradient(self, x, out = None):
        
        res = x.clone()

        norm = x.pnorm(2)
        norm_arr = norm.as_array().flatten()

        for el in res:
            el_arr = el.as_array()
            _get_gradient_numba(el_arr, norm_arr, self.theta)
        if out is None:
            return res

        else:
            out = res   

    def proximal(self, x, tau, out = None):
        
        res = x.clone()

        for el in res:
            el_arr = el.as_array()
            _get_proximal_numba(el_arr, tau, self.theta)

        if out is None:
            return res

        else:
            out = res   

    def convex_conjugate(self, x, out = None):
        res = x.clone()

        for el in res:
            el_arr = el.as_array()
            _get_convex_conjugate_numba(el_arr, self.theta)
            el.fill(el_arr)

        if out is None:
            return res.sum()

        else:
            out = res.sum()
@jit
def _get_value_numba(array, norm_array, theta):
    
    tmp = array.ravel()
    norm_array = norm_array.ravel()
    for i in prange(tmp.size):
        tmp[i] = theta*(norm_array[i]/theta-np.log(1+norm_array[i]/theta))

@jit    
def _get_gradient_numba(array, norm_array, theta):
    
    tmp = array.ravel()
    norm_array = norm_array.ravel()
    for i in prange(tmp.size):
        tmp[i] /= (1 + norm_array[i]/theta)
        tmp[i] /= 2
        
@jit    
def _get_proximal_numba(array, theta, tau):
    
    tmp = array.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] > 0:
            tmp[i] = np.sqrt((-theta - tau + tmp[i])**2 + 4*theta*tmp[i]) + -theta - tau + tmp[i]
        else:
            tmp[i] = -np.sqrt((-theta - tau - tmp[i])**2 - 4*theta*tmp[i]) + theta + tau + tmp[i]
            
        tmp[i]/=2
        
@jit    
def _get_convex_conjugate_numba(array, theta):
    
    tmp = array.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] > 0:
            x = tmp[i]/(tmp[i]+1)
            tmp[i] = theta*(x*tmp[i] - (np.abs(x) - np.log(1 + np.abs(x))))
        else:
            x = tmp[i]/(tmp[i]-1)
            tmp[i] = theta*(-x*tmp[i] - (np.abs(x) - np.log(1 + np.abs(x))))
    
        
