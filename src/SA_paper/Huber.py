from cil.optimisation.functions import Function

import numpy as np

from numba import jit, prange

@jit(nopython=True)
def _get_value_numba(arr, theta):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] > np.sqrt(2)*theta:
            tmp[i] -= theta/np.sqrt(2)
        else: 
            tmp[i] **= 2
            tmp[i] /= (2 * np.sqrt(2))
            tmp[i] /= theta
   
@jit(nopython=True)  
def _get_proximal_numba(arr, norm_arr, theta, tau):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        if norm_arr[i] > (np.sqrt(2)*theta+tau):
            tmp[i] *= np.maximum(0, 1 - tau/norm_arr[i])
            
        else:
            tmp[i] /= (1 + tau/(theta*np.sqrt(2)))
    
@jit(nopython=True)  
def _get_gradient_numba(arr, theta):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] > np.sqrt(2)*theta:
            continue
        else:
            tmp[i] = np.sqrt(2)

class Huber_SA(Function):
    
    """                 
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, theta, **kwargs):

        super(Huber_SA, self).__init__(L=1/theta)  
        self.theta = theta

    def __call__(self, x, out= None):
        norm = x.pnorm()
        norm_arr = norm.as_array()

        _get_value_numba(norm_arr, self.theta)

        if out is None:
            return np.sum(norm_arr)

        else:
            out = np.sum(norm_arr)

    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """      

        norm = x.pnorm()
        norm_arr = norm.as_array().flatten()
        res = x.clone()
        
        for el in res:
            el_arr = el.as_array()
            _get_proximal_numba(el_arr, norm_arr, self.theta, tau)
            el.fill(el_arr)
        
        if out is None:
            return res

        else:
            out = res   

    def gradient(self, x, out = None):
        
        denom = x.pnorm()
        norm_arr = denom.as_array()

        _get_gradient_numba(norm_arr, self.theta)
        denom.fill(norm_arr)
        
        if out is None:
            return x.divide(denom)
        else:
            x.divide(denom, out=out)   

    def get_gradient(self, x, out = None):
        if out is None:
            return self.gradient(self, x)

        else:
            self.gradient(self, x, out = out)

    def convex_conjugate(self, x, out = None):
        return np.inf