from cil.optimisation.functions import MixedL21Norm, SmoothMixedL21Norm, OperatorCompositionFunction
from cil.optimisation.operators import GradientOperator

from src.SA_paper import Huber_SA, Fair_SA, Charbonnier_SA
from src.Functions import create_sample_image, zoom_image, crop_image

from sirf.STIR import MessageRedirector, ImageData, AcquisitionData

from pytest import approx

msg_red = MessageRedirector('msg/info.txt', 'msg/warnings.txt', 'msg/errors.txt')

### parameters ###
theta = 0.0001 # small to mimick TV
tau = 0.0001 
alpha = 10
image_size = 2

# create sample image
templ_sino = AcquisitionData("data/template_sinogram.hs")
image = templ_sino.create_uniform_image()
image = zoom_image(image, image_size/image.dimensions()[1])
image = crop_image(templ_sino, image=image, nx = image_size, ny = image_size, nz = 1)
create_sample_image(image)

# gradient operator for TV
grad = GradientOperator(image, backend = 'numpy')
grad_image = grad.direct(image)

# create TV-like functions
huber = alpha * Huber_SA(theta)
charbonnier = alpha * Charbonnier_SA(theta)
fair = alpha * Fair_SA(theta)
tv = alpha * MixedL21Norm()
tv_smooth = alpha * SmoothMixedL21Norm(theta)


def test_always_passes():
    assert True
    
def grad_func(f1, f2):
    ocf1 = OperatorCompositionFunction(f1, grad)
    ocf2 = OperatorCompositionFunction(f2, grad)
    t1 = ocf1.gradient(image).as_array()
    t2 = ocf2.gradient(image).as_array()
    assert  t1 == approx(t2, abs=1e-8)
        
def prox_func(f1, f2):
    t1 = grad.adjoint(f1.proximal(grad_image, tau)).as_array()
    t2 = grad.adjoint(f2.proximal(grad_image, tau)).as_array()
    assert t1 == approx(t2, abs=1e-8)

def test_grad():        
    grad_func(huber, tv_smooth)
    grad_func(charbonnier, tv_smooth)
    grad_func(fair, tv_smooth)

def test_prox():
    prox_func(huber, tv)
    prox_func(charbonnier, tv)
    prox_func(fair, tv)
