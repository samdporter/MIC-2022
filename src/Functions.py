import numpy as np

from sirf.STIR import EllipticCylinder, TruncateToCylinderProcessor, ImageData
import sirf.Reg as Reg

def create_sample_image(image):
    '''fill the image with some simple geometric shapes.'''
    im_shape = image.shape
    image.fill(0)
    # create a shape
    shape = EllipticCylinder()
    shape.set_length(400)
    shape.set_radii((im_shape[1]//3*2, im_shape[2]//4*3))
    shape.set_origin((0, 0, 0))

    # add the shape to the image
    image.add_shape(shape, scale = 1)

    # add another shape
    shape.set_radii((im_shape[1]//2, im_shape[2]//4))
    shape.set_origin((0,0, -im_shape[2]//3))
    image.add_shape(shape, scale = -0.5)

    # add another shape
    shape.set_origin((0, 0, im_shape[2]//3))
    image.add_shape(shape, scale = -0.5)

    # add another shape
    shape.set_radii((im_shape[1]//8, im_shape[2]//8))
    shape.set_origin((0, im_shape[1]//5*2, 0))
    image.add_shape(shape, scale = 0.5)

    # add another shape
    shape.set_radii((im_shape[1]//10, im_shape[2]//10))
    shape.set_origin((0, -im_shape[1]//10, im_shape[2]//4))
    image.add_shape(shape, scale = 0.75)

def make_cylindrical_FOV(image):
    """truncate to cylindrical FOV."""
    cyl_filter = TruncateToCylinderProcessor()
    cyl_filter.apply(image)
    return image

def add_noise(proj_data,noise_factor = 0.1):
    """Add Poission noise to acquisition data."""
    proj_data_arr = proj_data.as_array() / noise_factor
    # Data should be >=0 anyway, but add abs just to be safe
    noisy_proj_data_arr = np.random.poisson(proj_data_arr).astype('float32');
    noisy_proj_data = proj_data.clone()
    noisy_proj_data.fill(noisy_proj_data_arr*noise_factor);
    return noisy_proj_data

def crop_image(templ_sino, image, nx, ny, nz, slice = None):
    """Crop from (vol_z,vol_y,vol_x) to (nz,ny,nx) and save to file"""
    vol = image.as_array()
    vol_dims = vol.shape
    x_origin = vol_dims[2]//2
    y_origin = vol_dims[1]//2
    if slice is None:
        z_origin = vol_dims[0]//2
    else:
        z_origin = slice
    
    vol = vol[z_origin-nz//2:z_origin+nz//2+nz%2,y_origin-ny//2:y_origin+ny//
              2+ny%2,x_origin-nx//2:x_origin+nx//2+nx%2]
    im = ImageData(templ_sino)
    dim=(nz,ny,nx)
    vol = vol.reshape(dim)
    voxel_size=im.voxel_sizes()
    im.initialise(dim,voxel_size)
    im.fill(vol)
    return im

def zoom_image(image, zoom):
    tm_identity = np.array([[1/zoom,0,0,0],
                        [0,1/zoom,0,0],
                        [0,0,1/zoom,0],
                        [0,0,0,1/zoom]])
    TM = Reg.AffineTransformation(tm_identity)
    resampler = Reg.NiftyResample()
    resampler.set_reference_image(image)
    resampler.set_floating_image(image)
    resampler.add_transformation(TM)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    resampler.process()
    return (resampler.get_output())