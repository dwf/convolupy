# NumPy/SciPy imports
import numpy as np
from numpy import random

# Local imports
from convolupy.base import BaseBPropComponent
from convolupy.constants import TANH_INNER, TANH_OUTER


class TanhSigmoid(BaseBPropComponent):
    """
    A backpropagation module that simply applies an elementwise
    sigmoid (hyperbolic tangent flavour).
    """
    def __init__(self, size, bias=False, nparams=None, params=None, 
                 grad=None, inner=TANH_INNER, outer=TANH_OUTER):
        """
        Construct this learning module.
        
        * size -- indicates the size of the image to which this sigmoid will
                  be applied.
        
        * inner/outer -- are values that the output are scaled by before
                         and after the squashing function is applied,
                         respectively.
                         
        * bias -- Initializes and uses a bias parameter in this plane.
                  Useful when you have several ConvolutionalPlanes feeding
                  into a single set of sigmoids and don't want each of those
                  linear layers learning a separate bias.
        """
        if nparams is None:
            nparams = 1 if bias else None
        super(TanhSigmoid, self).__init__(
            nparams=nparams, 
            params=params, 
            grad=grad
        )
        self.inner = inner
        self.outer = outer
        self._out_array = np.empty(size)
        self._bprop_array = np.empty(size)
        self.bias = self.params if self.params is None else self.params[0:1]
    
    def initialize(self, fan_in=None):
        """Initialize the bias parameter."""
        if fan_in is None:
            fan_in = 10 # There really is no good default value.
        std = fan_in**-0.5
        if self.params is None:
            raise ValueError('No parameters in this module!')
        size = self.params.shape
        self.params[:] = random.uniform(low=-2.4*std, high=2.4*std, size=size)
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        out = self._out_array
        if out is not inputs:
            out[...] = inputs
        if self.bias is not None:
            out += self.bias
        out *= self.inner
        np.tanh(out, out)
        out *= self.outer
        return out
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        out = self._bprop_array
        out[...] = inputs
        if self.bias is not None:
            out += self.bias
        out *= self.inner
        np.cosh(out, out)
        out **= -2.
        out *= self.outer * self.inner
        out *= dout
        return out
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        if self._grad is not None:
            # Dispatch this specifically to the TanhSigmoid class so that
            # it doesn't call overridden bprop's.
            self._grad[0] = TanhSigmoid.bprop(self, dout, inputs).sum()
        return self._grad
    

