import numpy as np
from numba import prange

def _get_value_numba(array, norm_array, theta):
    
    tmp = array.ravel()
    
    for i in prange(tmp.size):
        tmp[i] = theta*(norm_array[i]/theta-np.log(1+norm_array[i]/theta))
        
def _get_gradient_numba(array, norm_array, theta):
    
    tmp = array.ravel()
    norm_array = norm_array.ravel()
    for i in prange(tmp.size):
        tmp[i] /= (1 + norm_array[i]/theta)
        tmp[i] /= 2
        
def _get_proximal_numba(array, theta, tau):
    
    tmp = array.ravel()
    
    for i in prange(tmp.size):
        if tmp[i] > 0:
            tmp[i] = np.sqrt((-theta - tau + tmp[i])**2 + 4*theta*tmp[i]) + -theta - tau + tmp[i]
        else:
            tmp[i] = -np.sqrt((-theta - tau - tmp[i])**2 - 4*theta*tmp[i]) + theta + tau + tmp[i]
            
        tmp[i]/=2
        
            