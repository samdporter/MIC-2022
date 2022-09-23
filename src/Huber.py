from cil.optimisation.functions import Function
from sirf.STIR import Prior, try_calling
from sirf.Utilities import check_status
import sirf.pystir as pystir
import sirf.pyiutilities as pyiutil
import sirf.STIR_params as parms

from cil.optimisation.functions import MixedL21Norm, ConstantFunction, SumScalarFunction, SmoothMixedL21Norm

import numpy as np
import numba

@numba.jit(nopython=True)
def _get_proximal_numba(arr, norm, abstau, theta):
    '''Numba implementation of a step in the calculation of the proximal of Huber
    
    Parameters:
    -----------
    arr : numpy array, best if contiguous memory. 
    abstau: float >= 0
    Returns:
    --------
    Stores the output in the input array.
    Note:
    -----
    
    Input arr should be contiguous for best performance'''
    tmp = arr.ravel()
    denom = norm.flatten()
    for i in numba.prange(tmp.size):
        if denom[i] == 0:
            continue
        if denom[i] <= theta:
            tmp[i] /= ((abstau/theta)+1)
        else:
            tmp[i] *= (np.maximum(0, (1-(abstau)/denom[i])))

    return 0

class Huber(Function):
    
    """                 
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, theta, **kwargs):

        super(Huber, self).__init__(L=1/theta)  
        self.theta = theta

    def __call__(self, x, out= None):
        norm = x.pnorm()
        norm_arr = norm.as_array()

        for i in norm_arr.ravel():
            if i <= self.theta:
                i = i**2/(2*self.theta)
            else: 
                i -= self.theta/2

        if out is None:
            return np.sum(norm_arr)

        else:
            out = np.sum(norm_arr)

    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """      

        res = x.clone()
        denom = x.pnorm(2)
        for el in res:

            array = el.as_array()

            _get_proximal_numba(array, denom.as_array().flatten(), tau, self.theta)
            el.fill(array)

        if out is None:
            return res
        
        else:
            out = res 

    def gradient(self, x, out = None):
        norm = x.pnorm()
        norm_arr = norm.as_array().flatten()
        res = x.clone()

        for el in res:
            el_arr = el.as_array()
            for count, i in enumerate(el_arr):
                if norm_arr.ravel()[count] <= self.theta:
                    i /= self.theta
                else:
                    i /= norm_arr[count]
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
    
    
class Huber_CIL(Function):
    
    def __init__(self, theta, **kwargs):

        super(Huber_CIL, self).__init__(L=1/theta)  
        self.theta = theta
        
        self.SML21N = SumScalarFunction(SmoothMixedL21Norm(epsilon = 0), -theta/2)
        self.ML21N = SumScalarFunction(MixedL21Norm(), -theta/2)
        self.ML21NS = (1/theta) * MixedL21NormSquared()

    def __call__(self, x, out= None):
        
        norm = x.pnorm()
        norm_arr = norm.as_array()
        
        g_theta = norm.clone()
        l_theta = norm.clone()
        
        g_theta_arr = np.where(norm_arr > self.theta, norm_arr, 0)
        l_theta_arr = np.where(norm_arr <= self.theta, norm_arr, 0)
        
        g_theta = g_theta.fill(g_theta_arr) * x.clone()
        l_theta = l_theta.fill(l_theta_arr) * x.clone()
        
        res = self.ML21N(g_theta) + self.ML21NS(l_theta)
        
        if out is None:
            return res
    
        else:
            out = res

    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """      

        norm = x.pnorm()
        norm_arr = norm.as_array()
        
        g_theta = norm.clone()
        l_theta = norm.clone()
        
        g_theta_arr = np.where(norm_arr > self.theta, norm_arr, 0)
        l_theta_arr = np.where(norm_arr <= self.theta, norm_arr, 0)
        
        g_theta = g_theta.fill(g_theta_arr) * x.clone()
        l_theta = l_theta.fill(l_theta_arr) * x.clone()
        
        res = self.ML21N.proximal(g_theta, tau) + self.ML21NS.proximal(l_theta, tau)
        
        if out is None:
            return res
    
        else:
            out = res

    def gradient(self, x, out = None):
        norm = x.pnorm()
        norm_arr = norm.as_array()
        
        g_theta = norm.clone()
        l_theta = norm.clone()
        
        g_theta_arr = np.where(norm_arr > self.theta, norm_arr, 0)
        l_theta_arr = np.where(norm_arr <= self.theta, norm_arr, 0)
        
        g_theta = g_theta.fill(g_theta_arr) * x.clone()
        l_theta = l_theta.fill(l_theta_arr) * x.clone()
        
        res = self.SML21N.gradient(g_theta) + self.ML21NS.gradient(l_theta)
        
        if out is None:
            return res
    
        else:
            out = res  

    def convex_conjugate(self, x, out = None):
        return np.inf

class MixedL21NormSquared(Function):
    
    r""" L2NormSquared function: :math:`F(x) = \| x\|^{2}_{2} = \underset{i}{\sum}x_{i}^{2}`
          
    Following cases are considered:
                
        a) :math:`F(x) = \|x\|^{2}_{2}`
        b) :math:`F(x) = \|x - b\|^{2}_{2}`
        
    .. note::  For case b) case we can use :code:`F = L2NormSquared().centered_at(b)`,
               see *TranslateFunction*.
        
    :Example:
        
        >>> F = L2NormSquared()
        >>> F = L2NormSquared(b=b) 
        >>> F = L2NormSquared().centered_at(b)
                                                          
    """    
    
    def __init__(self, **kwargs):
        '''creator

        Cases considered (with/without data):            
                a) .. math:: f(x) = \|x\|^{2}_{2} 
                b) .. math:: f(x) = \|\|x - b\|\|^{2}_{2}

        :param b:  translation of the function
        :type b: :code:`DataContainer`, optional
        '''                        
        super(MixedL21NormSquared, self).__init__(L = 2)
        
                            
    def __call__(self, x):

        r"""Returns the value of the L2NormSquared function at x.
        
        Following cases are considered:
            
            a) :math:`F(x) = \|x\|^{2}_{2}`
    
        :param: :math:`x`
        :returns: :math:`\underset{i}{\sum}x_{i}^{2}`
                
        """          

        return ((1./2.) * x.pnorm(2).power(2)).sum()
                
    def gradient(self, x, out=None):        
        
        r"""Returns the value of the gradient of the L2NormSquared function at x.
        
        Following cases are considered:
                
            a) :math:`F'(x) = 2x`
                
        """
                
        if out is not None:
            
            out.fill(x)
            
        else:
            return x
        
                                                       
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the L2NormSquared function at x.
        
        Consider the following cases:
                
                a) .. math:: F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} 
                
        """                
            
        return ((1./2.) * x.pnorm(2).power(2)).sum()


    def proximal(self, x, tau, out = None):
        
        r"""Returns the value of the proximal operator of the L2NormSquared function at x.
        
        
        Consider the following cases:
                
                a) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x}{1+2\tau}
                b) .. math:: \mathrm{prox}_{\tau F}(x) = \frac{x-b}{1+2\tau} + b      
                        
        """            

        if out is None:
            
            return x/(1+tau)

        else:
            x.divide((1+tau), out=out)