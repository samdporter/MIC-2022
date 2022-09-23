from cil.optimisation.functions import Function

import numpy as np

from numba import jit, prange

@jit(nopython=True)
def _get_value_numba(arr,theta):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        tmp[i] = theta*(tmp[i]/theta - np.log(1 + tmp[i]/theta))
        
@jit(nopython=True)
def _get_proximal_numba(arr, norm_arr, theta, tau):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] >= 0:
            tmp[i] = np.sqrt((tmp[i] - theta - tau)**2 + 4 * theta * tmp[i]) \
                + tmp[i] - theta - tau
        else:
            tmp[i] = -np.sqrt((-tmp[i] - theta - tau)**2 - 4 * theta * tmp[i]) \
                + tmp[i] + theta + tau
        tmp[i] /= 2
        
@jit(nopython=True)
def _get_gradient_numba(arr, theta):
    
    tmp = arr.ravel()
    
    for i in prange(tmp.size):
        tmp[i] = theta + tmp[i]

class Fair_SA(Function):
    
    """                 
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, theta, **kwargs):

        super(Fair_SA, self).__init__(L=1/theta)  
        self.theta = theta

    def __call__(self, x, out= None):
        
        res = x.pnorm()
        norm_arr = res.as_array()
        
        _get_value_numba(norm_arr, self.theta)
        res.fill(norm_arr)
        
        if out is None:
            return res.sum()

        else:
            out = res.sum()

    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """      
        res = x.clone()
        norm_arr = x.pnorm().as_array().flatten()
        
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
    
class Fair_SA_no_numba(Function):
    
    """                 
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, theta, **kwargs):

        super(Fair_SA_no_numba, self).__init__(L=1/theta)  
        self.theta = theta

    def __call__(self, x, out= None):
        
        res = x.clone()
        norm_arr = x.pnorm().as_array().flatten()
        
        
        for el in res:
            el_arr = el.as_array()
            tmp = el_arr.ravel()
            for i in range(tmp.size):
                tmp[i] = self.theta**2 * (norm_arr[i] / self.theta - np.log(1 + norm_arr[i] / self.theta))
            el.fill(el_arr)
        if out is None:
            return res.sum()

        else:
            out = res.sum()

    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """      
        res = x.clone()
        norm_arr = x.pnorm().as_array().flatten()
        
        for el in res:
            el_arr = el.as_array()
            tmp = el_arr.ravel()
            for i in range(tmp.size):
                if tmp[i] > 0:
                    tmp[i] = np.sqrt((tmp[i] - self.theta - self.theta * tau)**2 + 4 * self.theta * tmp[i]) \
                        + tmp[i] - self.theta - self.theta * tau
                else:                    
                    tmp[i] = np.sqrt((-tmp[i] - self.theta - self.theta * tau)**2 - 4 * self.theta * tmp[i]) \
                        + tmp[i] + self.theta + self.theta * tau
                        
                tmp[i] /= 2
            el.fill(el_arr)
        
        if out is None:
            return res

        else:
            out = res   

    def gradient(self, x, out = None):
        res = x.clone()
        norm_arr = x.pnorm().as_array().flatten()

        for el in res:
            el_arr = el.as_array()
            tmp = el_arr.ravel()
            for i in range(tmp.size):
                 tmp[i] /= np.sqrt(1 + norm_arr[i]/self.theta)   
            el.fill(el_arr)
        
        if out is None:
            return res

        else:
            out = res   

    def get_gradient(self, x, out = None):
        if out is None:
            return self.gradient(self, x)

        else:
            self.gradient(self, x, out = out)

    def convex_conjugate(self, x, out = None):
        return np.inf