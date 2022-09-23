from re import S
from cil.optimisation.algorithms import Algorithm
import warnings

class ADMM(Algorithm):
        
    ''' 
        ADMM is the Alternating Direction Method of Multipliers (ADMM)
    
        General form of ADMM : min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        The quadratic term in the augmented Lagrangian is linearized for the x-update.
                            
        Main algorithmic difference is that in ADMM we compute two proximal subproblems, 
        where in the PDHG a proximal and proximal conjugate.

            x^{k} = prox_{\tau f } (x^{k-1} - A^{T}u^{k-1} )                
            
            z^{k} = prox_{\sigma g} (Ax^{k} + u^{k-1})
            
            u^{k} = u^{k-1} + Ax^{k} - z^{k}
                
    '''

    def __init__(self, f=None, g=None, operator=None, \
                       tau = None, sigma = 1., \
                       initial = None, project = True, \
                       accelerate = True, r = 3, \
                       relax = True, **kwargs):
        
        '''Initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal
        :param g: Convex function with "simple" proximal 
        :param sigma: Positive step size parameter 
        :param tau: Positive step size parameter
        :param initial: Initial guess ( Default initial_guess = 0)'''        
        
        super(ADMM, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = kwargs.get('use_axpby', True)

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial, project = project,\
             accelerate = accelerate, r = r, relax = relax, **kwargs)        
                    
    def set_up(self, f, g, operator, tau = None, sigma=1., \
        initial=None, project = True, accelerate = True, r= 3, \
        relax = False, **kwargs):

        print("{} setting up".format(self.__class__.__name__, ))
        
        if sigma is None and tau is None:
            raise ValueError('Need tau <= sigma / ||K||^2')

        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma
        self.r = r
        self.accelerate = accelerate

        self.project = project
        
        self.relax = relax

        if self.tau is None:
            normK = self.operator.norm()
            self.tau = self.sigma / normK ** 2
            
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
            self.z = self.operator.range_geometry().allocate(0)
        else:
            self.x = initial.copy()  
            self.z = self.operator.direct(self.x)

        self. u = self.operator.range_geometry().allocate(0)
        
        if self.accelerate:
            self.u_old = self.u.clone()
            self.u_bar = self.u.clone()
            self.z_old = self.z.clone()
            

        self.configured = True  
        
        print("{} configured".format(self.__class__.__name__, ))
        
    def update(self):
        r""" Performs a single iteration of the ADMM algorithm"""

        self.x = self.f.proximal(self.x - self.operator.adjoint(self.u), self.tau)

        self.z = self.g.proximal(self.operator.direct(self.x) + self.u, self.sigma)

        # update_previous_solution() called after update by base class
        #i.e current solution is now in x_old, previous solution is now in x   
         
        if self.project:
            self.project_to_positive()
            
        if self.accelerate:
            gamma = self.iteration/(self.iteration + self.r)
            self.u_bar = self.u + self.operator.direct(self.x) -self.z
            self.u = self.u_bar + gamma*(self.u_bar-self.u_old)
            self.z += gamma*(self.z-self.z_old)
        else:
            self.u += self.operator.direct(self.x)
            self.u -= self.z
        
        if self.relax:
            self.relax_step()
            
    def project_to_positive(self):
        self.x.add(self.x.abs(), out = self.x) ## new line
        self.x.divide(2, out = self.x)    ## new line  
        
    def relax_step(self):
        # try clause to prevent error on first iter
        try:
            if self.loss[-2] < self.loss[-1]:
                self.tau /= 2
                self.sigma/=2
            else:
                self.tau *=1.01
                self.sigma *= 1.01
        except: return

    def update_objective(self):
        
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) )  


class LADMM(Algorithm):
        
    ''' 
        ADMM is the Alternating Direction Method of Multipliers (ADMM)
    
        General form of ADMM : min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        The quadratic term in the augmented Lagrangian is linearized for the x-update.
                            
        Main algorithmic difference is that in ADMM we compute two proximal subproblems, 
        where in the PDHG a proximal and proximal conjugate.

            x^{k} = prox_{\tau f } (x^{k-1} - A^{T}u^{k-1} )                
            
            z^{k} = prox_{\sigma g} (Ax^{k} + u^{k-1})
            
            u^{k} = u^{k-1} + Ax^{k} - z^{k}
                
    '''

    def __init__(self, f=None, g=None, operator=None, \
                       tau = None, sigma = 1., \
                       initial = None, project = True, \
                       accelerate = True, r = 3, \
                       relax = True, **kwargs):
        
        '''Initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal
        :param g: Convex function with "simple" proximal 
        :param sigma: Positive step size parameter 
        :param tau: Positive step size parameter
        :param initial: Initial guess ( Default initial_guess = 0)'''        
        
        super(LADMM, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = kwargs.get('use_axpby', True)

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial, project = project,\
             accelerate = accelerate, r = r, relax = relax, **kwargs)        
                    
    def set_up(self, f, g, operator, tau = None, sigma=1., \
        initial=None, project = True, accelerate = True, r= 3, \
        relax = False, **kwargs):

        print("{} setting up".format(self.__class__.__name__, ))
        
        if sigma is None and tau is None:
            raise ValueError('Need tau <= sigma / ||K||^2')

        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma
        self.r = r
        self.accelerate = accelerate

        self.project = project
        
        self.relax = relax

        if self.tau is None:
            normK = self.operator.norm()
            self.tau = self.sigma / normK ** 2
            
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
            self.z = self.operator.range_geometry().allocate(0)
        else:
            self.x = initial.copy()  
            self.z = self.operator.direct(self.x)

        self. u = self.operator.range_geometry().allocate(0)
        
        if self.accelerate:
            self.u_old = self.u.clone()
            self.u_bar = self.u.clone()
            self.z_old = self.z.clone()
            

        self.configured = True  
        
        print("{} configured".format(self.__class__.__name__, ))
        
    def update(self):
        r""" Performs a single iteration of the ADMM algorithm"""

        self.x = self.f.proximal(self.x - self.operator.adjoint(self.u), self.tau)

        self.z = self.g.proximal(self.operator.direct(self.x) + self.u, self.sigma)

        # update_previous_solution() called after update by base class
        #i.e current solution is now in x_old, previous solution is now in x   
         
        if self.project:
            self.project_to_positive()
            
        if self.accelerate:
            gamma = self.iteration/(self.iteration + self.r)
            self.u_bar = self.u + self.operator.direct(self.x) -self.z
            self.u = self.u_bar + gamma*(self.u_bar-self.u_old)
            self.z += gamma*(self.z-self.z_old)
        else:
            self.u += self.operator.direct(self.x)
            self.u -= self.z
        
        if self.relax:
            self.relax_step()
            
    def project_to_positive(self):
        self.x.add(self.x.abs(), out = self.x) ## new line
        self.x.divide(2, out = self.x)    ## new line  
        
    def relax_step(self):
        # try clause to prevent error on first iter
        try:
            if self.loss[-2] < self.loss[-1]:
                self.tau /= 2
                self.sigma/=2
            else:
                self.tau *=1.005
                self.sigma *= 1.005
        except: return

    def update_objective(self):
        
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) ) 

class SimpleADMM(Algorithm):
        
    ''' 
        ADMM is the Alternating Direction Method of Multipliers (ADMM)
    
        General form of ADMM : min_{x} f(x) + g(y), subject to Ax + By = b
    
        Case: A = Id, B = -K, b = 0   ==> min_x f(Kx) + g(x)  
                
        The quadratic term in the augmented Lagrangian is linearized for the x-update.
                            
        Main algorithmic difference is that in ADMM we compute two proximal subproblems, 
        where in the PDHG a proximal and proximal conjugate.

            x^{k} = prox_{\tau f } (x^{k-1} - A^{T}u^{k-1} )                
            
            z^{k} = prox_{\sigma g} (Ax^{k} + u^{k-1})
            
            u^{k} = u^{k-1} + Ax^{k} - z^{k}
                
    '''

    def __init__(self, f=None, g=None, operator=None, \
                       tau = None, sigma = 1., \
                       initial = None, project = True, \
                       accelerate = True, r = 3, \
                       relax = True, **kwargs):
        
        '''Initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal
        :param g: Convex function with "simple" proximal 
        :param sigma: Positive step size parameter 
        :param tau: Positive step size parameter
        :param initial: Initial guess ( Default initial_guess = 0)'''        
        
        super(SimpleADMM, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = kwargs.get('use_axpby', True)

        self.set_up(f = f, g = g, operator = operator, tau = tau,\
             sigma = sigma, initial=initial, project = project,\
             accelerate = accelerate, r = r, relax = relax, **kwargs)        
                    
    def set_up(self, f, g, operator, tau = None, sigma=1., \
        initial=None, project = True, accelerate = True, r= 3, \
        relax = True, **kwargs):

        print("{} setting up".format(self.__class__.__name__, ))
        
        if sigma is None and tau is None:
            raise ValueError('Need tau <= sigma / ||K||^2')

        self.f = f
        self.g = g
        self.operator = operator

        self.tau = tau
        self.sigma = sigma
        self.r = r
        self.accelerate = accelerate

        self.project = project
        
        self.relax = relax

        if self.tau is None:
            normK = self.operator.norm()
            self.tau = self.sigma / normK ** 2
            
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
            self.z = self.operator.range_geometry().allocate(0)
        else:
            self.x = initial.copy()  
            self.z = self.operator.direct(self.x)

        self. u = self.operator.range_geometry().allocate(0)

        self.configured = True  
        
        print("{} configured".format(self.__class__.__name__, ))
        
    def update(self):
        r""" Performs a single iteration of the ADMM algorithm"""

        self.x = self.f.proximal(self.operator.adjoint(self.z), self.tau)

        self.z = self.g.proximal(self.operator.direct(self.x), self.sigma)

        # update_previous_solution() called after update by base class
        #i.e current solution is now in x_old, previous solution is now in x   
         
        if self.project:
            self.project_to_positive()
            
    def project_to_positive(self):
        self.x.add(self.x.abs(), out = self.x) ## new line
        self.x.divide(2, out = self.x)    ## new line  

    def update_objective(self):
        
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) )

    def update_relaxation(self):
        self.step_size *= self.relax_coeff     