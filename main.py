import glob
import os
import pandas as pd

from sirf.STIR import EllipticCylinder, ImageData, AcquisitionData, \
    AcquisitionModelUsingRayTracingMatrix, MessageRedirector, \
    TruncateToCylinderProcessor, make_Poisson_loglikelihood, \
    OSMAPOSLReconstructor

import numpy as np

from cil.optimisation.functions import KullbackLeibler, SmoothMixedL21Norm, \
    OperatorCompositionFunction, IndicatorBox, BlockFunction, MixedL21Norm, \
    L2NormSquared, ConstantFunction
from cil.optimisation.algorithms import PDHG, GD, LADMM
from cil.optimisation.operators import GradientOperator, BlockOperator, IdentityOperator, \
    CompositionOperator, LinearOperator

from scipy.optimize import minimize, Bounds

import matplotlib.pyplot as plt
import seaborn as sns

from src.Huber import Huber, Huber_CIL, MixedL21NormSquared
from src.FairPotential import Fair
from src.ADMM import ADMM
from src.DirectionalTV import DirectionalTV
from src.SA_paper.Huber import Huber_SA
from src.SA_paper.Charbonnier import Charbonnier_SA
from src.SA_paper.Fair import Fair_SA

from src.StirPrior import StirPrior

from src.Functions import create_sample_image, make_cylindrical_FOV, add_noise, crop_image, zoom_image

from omegaconf import DictConfig, OmegaConf
import hydra

wd = os.getcwd()
print(wd)

msg_red = MessageRedirector(os.path.join(wd, 'msg/info.txt'), os.path.join(wd, 'msg/warnings.txt'), 
                            os.path.join(wd, 'msg/errors.txt'))

def calculate_norm_CO(self, **kwargs):
    '''Returns the norm of the Composition Operator

    if the operator in the block do not have method norm defined, i.e. they are SIRF
    AcquisitionModel's we use PowerMethod if applicable, otherwise we raise an Error
    '''
    norm = []
    for i, op in enumerate(self.operators):
        if hasattr(op, 'norm'):
            norm.append(op.norm(**kwargs) ** 2.)
        else:
            # use Power method
            if op.is_linear():
                norm.append(
                        LinearOperator.PowerMethod(op, 20)[0]
                        )
            else:
                raise TypeError('Operator {} does not have a norm method and is not linear'.format(op))
    return np.sqrt(sum(norm))

CompositionOperator.calculate_norm = calculate_norm_CO


@ hydra.main(config_path = "config", config_name="cfg")
def main(cfg: DictConfig) -> None:

    data_fidelity_choice = cfg.set_up.data_fidelity
    prior_choice = cfg.set_up.prior
    operator = cfg.set_up.operator
    warm_start = cfg.set_up.warm_start
    warm_start_iters = cfg.set_up.warm_start_iters

    alpha = cfg.hyperparameters.alpha
    theta = cfg.hyperparameters.theta
    epsilon = cfg.hyperparameters.epsilon
    
    iters = cfg.algorithm_settings.num_iters
    algorithm_dict = {}
    algorithm_dict["admm"] = cfg.algorithm_settings.admm
    algorithm_dict["ladmm"] = cfg.algorithm_settings.ladmm
    algorithm_dict["pdhg"] = cfg.algorithm_settings.pdhg
    algorithm_dict["bfgs"] = cfg.algorithm_settings.bfgs
    algorithm_dict["gdes"] = cfg.algorithm_settings.gdes
    algorithm_dict["osem"] = cfg.algorithm_settings.osem
        
    image_size = cfg.image_settings.image_size
    
    noise_level = cfg.data_settings.noise_level

    # create template acquisiton, phantom image and 
    templ_sino = AcquisitionData(os.path.join(wd,"data/template_sinogram.hs"))
    image = templ_sino.create_uniform_image()
    print(image.dimensions()[1])
    print(image_size/image.dimensions()[1])
    image = zoom_image(image, image_size/image.dimensions()[1])
    print(image.dimensions()[1])
    image = crop_image(templ_sino, image=image, nx = image_size, ny = image_size, nz = 1)
    create_sample_image(image)
    #image *= 250
    init = make_cylindrical_FOV(image.get_uniform_copy(image.max()/5))

    image.write("ground_truth")
    
    anatomical = image.clone()
    
    anatomical.write("anatomical")

    # create acquisition model
    acq_model = AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_background_term(templ_sino.clone().fill(epsilon))
    acq_model.set_up(templ_sino, image)
    acq_model_lin = acq_model.get_linear_acquisition_model()

    # create simple data
    acquired_data=acq_model.forward(image)
    noisy_data = add_noise(acquired_data, noise_factor = noise_level)

    # objective function to replace KL gradient (faster & more precise)
    obj_fun = make_Poisson_loglikelihood(noisy_data, acq_model=acq_model)
    obj_fun.set_up(image)

    def KL_gradient(self, x, out = None):
        if out is None:
            return -obj_fun.gradient(x)
        else:
            out = -obj_fun.gradient(x)

    #KullbackLeibler.CIL_gradient = KullbackLeibler.gradient
    #KullbackLeibler.gradient = KL_gradient

    # Data fidelity function
    if data_fidelity_choice == "Kullback Leibler":
        DF = KullbackLeibler(b=noisy_data, eta = templ_sino.clone().fill(epsilon))
    elif data_fidelity_choice == "Least Squares":
        DF = L2NormSquared(b=noisy_data)
    DF_ocf_am = OperatorCompositionFunction(DF, acq_model_lin)

    ### prior function ###
    if prior_choice == "Total Variation":
        prior = alpha * MixedL21Norm()
        prior_smooth = alpha * SmoothMixedL21Norm(epsilon = theta)
        
    elif prior_choice == "Fair":
        prior = alpha * Fair_SA(theta = theta)
        prior_smooth = prior
        
    elif prior_choice == "Huber":
        prior = alpha * Huber_CIL(theta = theta)
        prior_smooth = prior
    
    elif prior_choice == "Charbonnier":
        prior = alpha * Charbonnier_SA(theta = theta)
        prior_smooth = prior

    elif prior_choice == "None":
        prior = alpha * ConstantFunction()
        prior_smooth = prior
        
    elif prior_choice == "Quadratic":
        prior = alpha *  MixedL21NormSquared() ### for quadratic prior
        prior_smooth = prior
        
    if operator == "dTV":
        grad = CompositionOperator(DirectionalTV(anatomical), GradientOperator(image))
    elif operator == "identity":
        grad = IdentityOperator(image)
    else: grad = GradientOperator(image)

    # attempt to incorporate cil prior into sirf recon - not currently working
    prior_stir = StirPrior()
    prior_stir.set_up(prior = prior_smooth, theta = 1, image = image, alpha = alpha)

    # cil operator composition functions for prior - used in GD and BFGS
    prior_ocf_grad = OperatorCompositionFunction(prior, grad)
    try:
        prior_smooth_ocf_grad = OperatorCompositionFunction(prior_smooth, grad)
    except: print("No smooth prior available")

    # non-negativity constraint
    pos_con = IndicatorBox(0)

    # set up block geometry for algorithms
    bf = BlockFunction(DF, prior)
    bo = BlockOperator(acq_model_lin, grad)


    if warm_start:
        # warm start gradient projection for KL
        if data_fidelity_choice == "Kullback Leibler":
            # create BSREM trype preconditioner:
            sens = acq_model.backward(noisy_data.get_uniform_copy(1))
            preconditioner = init.clone().get_uniform_copy(1)
            step_size = 1
        elif data_fidelity_choice == "Least Squares":
            step_size = 0.01
        else:
            raise NotImplementedError

        step = step_size
        for _ in range(warm_start_iters):
            if data_fidelity_choice == "Kullback Leibler":
                preconditioner.fill(np.nan_to_num(((init+0.0001)/sens).as_array(), posinf=0, neginf=0))
                step = step_size * preconditioner
            init -= step*(DF_ocf_am.gradient(init))
            init += init.abs()
            init /=2

    init.write("init_image")
    
    # choose weighting between proximal steps
    g = init.norm()
    sigma = g
    tau = 1 / (g* bo.norm()**2)

    ### algorithms set up ###

    objectives_dict = {} # dictinary containing objective function valuies for algorithms
    recon_image_dict = {} # dictionary containing 
    recon_grad_image_dict = {} # dictionary containing objective function gradient images of reconstructed images

    ### OSEM ###
    if algorithm_dict["osem"]:
        obj_fun.set_prior(prior_stir)
        reconstructor = OSMAPOSLReconstructor()
        reconstructor.set_objective_function(obj_fun)
        reconstructor.set_num_subsets(21)
        reconstructor.set_num_subiterations(iters//2)
        reconstructor.set_input(noisy_data)
        reconstructor.set_up(init)
        OSEM_im = init.clone()
        reconstructor.reconstruct(OSEM_im)

        OSEM_im = reconstructor.get_current_estimate()
        reconstructor.set_num_subsets(1)
        reconstructor.set_num_subiterations(iters//2)
        reconstructor.set_input(noisy_data)
        reconstructor.set_up(init)
        reconstructor.reconstruct(OSEM_im)

        OSEM_soln = reconstructor.get_current_estimate()

        OSEM_soln.write("OSEM")

    ### ADMM ###
    sigma_admm = sigma
    tau_admm = tau/10
    if algorithm_dict["admm"]:

        admm = ADMM(g = bf, f = pos_con, operator = bo, initial = init, max_iteration = iters, \
                        sigma = sigma_admm/100/bo.norm()**2, tau = tau_admm/10)
        admm.run()
        
        objectives_dict["admm"] = admm.objective
        recon_image_dict["admm"] = admm.solution
        recon_grad_image_dict["admm"] = prior_smooth_ocf_grad.gradient(admm.solution) + \
            DF_ocf_am.gradient(admm.solution)
        admm.solution.write("admm")
    
        
    if algorithm_dict["ladmm"]:
        ladmm = LADMM(g = bf, f = pos_con, operator = bo, initial = init, max_iteration = iters, \
                        sigma = sigma_admm, tau = tau_admm)
        ladmm.run()
        
        objectives_dict["ladmm"] = ladmm.objective
        recon_image_dict["ladmm"] = ladmm.solution
        recon_grad_image_dict["ladmm"] = prior_smooth_ocf_grad.gradient(ladmm.solution) + \
            DF_ocf_am.gradient(ladmm.solution)
        ladmm.solution.write("ladmm")

    ### L-BFGS-B ###
    if algorithm_dict["bfgs"]:
        bounds = Bounds(lb = 0, ub = np.inf)
        bfgs_objective = []

        def obj_value_comp(x, f1, f2, image):
            tmp = image.copy()
            tmp.fill(x.reshape(image.shape))
            return f1.__call__(tmp) + f2.__call__(tmp)

        def obj_grad_comp(x, f1, f2, image):
            tmp = image.copy()
            tmp.fill(x.reshape(image.shape))
            gradient = f1.gradient(tmp) + f2.gradient(tmp)
            return np.float64(gradient.as_array().flatten())

        def callbackL(x):
            val = obj_value_comp(x, DF_ocf_am, prior_smooth_ocf_grad, init)
            bfgs_objective.append(val)
            print("Objective value: " + str(val))

        callbackL(init.as_array())

        bfgs = minimize(obj_value_comp, init.as_array().flatten(), \
                        args = (DF_ocf_am, prior_smooth_ocf_grad, init), method = "L-BFGS-B", \
                        callback = callbackL, bounds = bounds, jac=obj_grad_comp, tol = -1, \
                        options={'maxiter': iters, 'ftol':0.,'gtol':0., 'maxfun':1000000})

        bfgs_soln = init.clone().fill(bfgs.x.reshape(image.shape))
        
        objectives_dict["bfgs"] = bfgs_objective
        recon_image_dict["bfgs"] = bfgs_soln
        recon_grad_image_dict["bfgs"] = prior_smooth_ocf_grad.gradient(bfgs_soln) + \
            DF_ocf_am.gradient(bfgs_soln)
        bfgs_soln.write("bfgs")

    ### PDHG ###
    if algorithm_dict["pdhg"]:
        pdhg = PDHG(f = bf, g = pos_con, operator = bo, initial = init, max_iteration = iters,
                    sigma = sigma, tau = tau)
        
        pdhg.run()
        
        objectives_dict["pdhg"] = pdhg.objective
        recon_image_dict["pdhg"] = pdhg.solution
        recon_grad_image_dict["pdhg"] = prior_smooth_ocf_grad.gradient(pdhg.solution) + \
            DF_ocf_am.gradient(pdhg.solution)
        pdhg.solution.write("pdhg")

    ### Gradient Descent ###
    if algorithm_dict["gdes"]:
        gd_soln = init.clone() # copy init image to be used in gd recon

        # create BSREM trype preconditioner
        sens = acq_model.backward(noisy_data.get_uniform_copy(1))
        preconditioner = gd_soln.clone().get_uniform_copy(1)

        gd_objective = []
        step_size = 1
        for i in range(iters):
            #track objective function values
            val = DF_ocf_am(gd_soln) + prior_smooth_ocf_grad(gd_soln)
            print(val)
            gd_objective.append(val)

            # gradient descent
            preconditioner.fill(np.nan_to_num(((gd_soln+0.0001)/sens).as_array(), posinf=0, neginf=0))
            gd_soln -= step_size*preconditioner*(DF_ocf_am.gradient(gd_soln)+ prior_smooth_ocf_grad.gradient(gd_soln))
            # non-negativity constraint
            gd_soln += gd_soln.abs()
            gd_soln /=2

            # relaxation / mini line search
            try:
                if val >= gd_objective[-2]:
                    step_size /= 1.5
                else:
                    step_size *= 1.01
            except:
                print("first iter")

        gd_objective.append(DF_ocf_am(gd_soln)+ prior_smooth_ocf_grad(gd_soln))
        
        objectives_dict["gd"] = gd_objective
        recon_image_dict["gd"] = gd_soln
        recon_grad_image_dict["gd"] = prior_smooth_ocf_grad.gradient(gd_soln) + \
            DF_ocf_am.gradient(gd_soln)
        gd_soln.write("gd")
    ### Algorithms run ###

    # plot objectives for comparison
    plt.figure()
    max_val = 0
    for key in objectives_dict:
        max_val = np.maximum(max_val , np.max(objectives_dict[key]))
    for key in objectives_dict:   
        plt.plot(objectives_dict[key], label = key + str(objectives_dict[key][-1]) + " " + str(np.min(objectives_dict[key])))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("objectives")
    plt.show()

    for key in recon_grad_image_dict:
        recon_grad_image_dict[key].write(key + "_grad")

    max_obj_len = 0
    for key in objectives_dict:
        max_obj_len = np.maximum(max_obj_len, len(objectives_dict[key]))
    for key in objectives_dict:
        if len(objectives_dict[key]) < max_obj_len:
            objectives_dict[key] += (max_obj_len - len(objectives_dict[key]))*[np.nan]
        print(len(objectives_dict[key]))

    objectives_df = pd.DataFrame(objectives_dict)
    objectives_df.to_csv("objectives", index = False)
    
    #image_df = pd.DataFrame(recon_image_dict)
    #image_df.to_csv("images", index = False)

    for filename in glob.glob(os.path.join(wd, "tmp*")):
        os.remove(filename)
    for filename in glob.glob(os.path.join(os.getcwd(), "tmp*")):
        os.remove(filename)
    
    result = prior_smooth_ocf_grad(bfgs_soln) +  DF_ocf_am(bfgs_soln) - prior_smooth_ocf_grad(image) +  DF_ocf_am(image)
    
    
    return result

main()