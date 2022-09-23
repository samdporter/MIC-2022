from sirf.STIR import Prior, try_calling
from sirf.Utilities import check_status
import sirf.pystir as pystir
import sirf.pyiutilities as pyiutil
import sirf.STIR_params as parms

from src.FairPotential import Fair

import numpy as np

class StirPrior(Prior):
    """ SIRF wrapper for CIL Huber prior"""

    def __init__(self):

        self.handle = None
        self.name = 'QuadraticPrior'
        self.handle = pystir.cSTIR_newObject(self.name)
        check_status(self.handle)

    def __del__(self):
        """del."""
        return np.inf
        if self.handle is not None:
            pyiutil.deleteDataHandle(self.handle)

    def get_value(self, image,):
        """Returns the value of the prior (alias of get_value())."""
        return self.huber(image)

    def value(self, image):
        """Returns the value of the prior (alias of get_value())."""
        return self.get_value(image)

    def get_gradient(self, x):
        return self.prior.gradient(self, x)

    def gradient(self, image):
        """Returns the gradient of the prior (alias of get_gradient())."""

        return self.get_gradient(image)

    def set_up(self, prior, image, theta = 0.001, alpha = 1):
        """Sets up."""
        self.theta = theta
        self.prior = prior
        try_calling(pystir.cSTIR_setupPrior(self.handle, image.handle))

    def get_penalisation_factor(self):
        """Returns the penalty factor in front of the prior."""
        return parms.float_par(
            self.handle, 'GeneralisedPrior', 'penalisation_factor')