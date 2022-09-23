from cil.optimisation.operators import LinearOperator, GradientOperator
from cil.framework import BlockGeometry, ImageGeometry
import numpy as np

from numba import jit

@jit(nopython=True)
def array_dot(arr0, arr1):
    out_array = np.zeros(arr0[0].shape)
    for i in range(len(arr0)):
        out_array += arr0[i]*arr1[i]
    return out_array
    
def bdc_dot(bdc0, bdc1, image):
    arr_list0 = []
    arr_list1 = []
    for i in bdc0:
        arr_list0.append(np.squeeze(i.clone().as_array()))
    for j in bdc1:
        arr_list1.append(np.squeeze(j.clone().as_array()))
    arr = array_dot(np.array(arr_list0),np.array(arr_list1)).reshape(image.shape)
    return image.clone().fill(arr)


class DirectionalTV(LinearOperator):
    def __init__(self, anatomical_image, nu = 0.01, gamma=1, smooth = True, beta = 0.001,**kwargs):
        """Constructor method"""    
        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        self.is2D = False
        self.domain_shape = []
        self.ind = []
        if smooth is True:
            self.beta = beta
        self.voxel_size_order = []
        self._domain_geometry = anatomical_image
        for i, size in enumerate(list(self._domain_geometry.shape) ):
            if size!=1:
                self.domain_shape.append(size)
                self.ind.append(i)
                self.voxel_size_order.append(self._domain_geometry.spacing[i])
                self.is2D = True

        self.gradient = GradientOperator(anatomical_image, backend='numpy')
        
        self.anato = anatomical_image
    
        self.tmp_im = anatomical_image.clone()
        
       	# smoothing for xi 
        self.gamma = gamma
        
        self.anato_grad = self.gradient.direct(self.anato) # gradient of anatomical image

        self.denominator = (self.anato_grad.pnorm(2).power(2) + nu**2) #smoothed norm of anatomical image  

        self.ndim = len(self.domain_shape)

        super(DirectionalTV, self).__init__(BlockGeometry(*[self._domain_geometry for _ in range(self.ndim)]), 
              range_geometry = BlockGeometry(*[self._domain_geometry for _ in range(self.ndim)]))

    def direct(self, x, out=None): 
        inter_result = bdc_dot(x.clone(),self.anato_grad.clone(), self.tmp_im)

        if out is None:       
            return x.clone() - self.gamma*inter_result*(self.anato_grad/self.denominator) # (delv * delv dot del u) / (norm delv) **2
                
        else:
            out = x.clone() - self.gamma*inter_result*(self.anato_grad/self.denominator)
                
        
        
    def adjoint(self, x, out=None):  
        return self.direct(x, out=out) # self-adjoint operator


class igify(LinearOperator):
    def __init__(self, blockdatacontainer):
        
        self.BDC = blockdatacontainer
        self.vox_size = self.BDC[0].voxel_sizes()
        self.dims = self.BDC[0].dimensions()
        self.channels = len(self.BDC)
        self.ig = ImageGeometry(voxel_num_y=self.dims[1],
                     voxel_size_x=self.vox_size[2],
                     voxel_num_x=self.dims[2],
                     voxel_size_y=self.vox_size[1],
                     channels = self.channels)
        super(igify, self).__init__(domain_geometry=self.BDC, 
            range_geometry = self.ig)
        
    def direct(self, x, out = None):
        arrs = []
        for im in range(len(x)):
            arrs.append(np.squeeze(x[im].as_array()))
        res = self.ig.allocate()
        res.fill(np.array(arrs))
        if out is None:
            return res
        else:
            out = res
            
    def adjoint(self, x, out = None):
        arr = x.as_array()
        res = self.BDC.clone()
        for i in range(self.channels):
            res[i].fill(arr[i].reshape((1,self.dims[1],self.dims[2])))
        if out is None:
            return res
        else:
            out = res