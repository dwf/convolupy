"""
Module for feature maps: classes that combine inputs and pass through a 
squashing function.
"""
# Standard library imports
from itertools import izip

# NumPy/SciPy imports
import numpy as np
from numpy import random

# Local imports
from convolupy.base import BaseBPropComponent
from convolupy.sigmoids import TanhSigmoid
from convolupy.planes import ConvolutionalPlane, AveragePoolingPlane
from convolupy.constants import TANH_INNER, TANH_OUTER

class NaiveConvolutionalFeatureMap(BaseBPropComponent):
    """
    One way to implement a standard feature map that takes input from a 
    single lower-level image. This serves two purposes: to demonstrate 
    how to write new learning modules by composing two existing modules,
    and to serve as a sanity check for the more efficient implementation,
    ConvolutionalFeatureMap.
    
    Has, as members, a ConvolutionalPlane with standard bias configuration
    and a TanhSigmoid object that does the squashing.
    
    This is a little wasteful since each of the modules has separate output
    array members. See FeatureMap for a slightly more memory efficient 
    implementation that uses subclassing.
    """
    def __init__(self, fsize, imsize):
        """Construct a feature map with given filter size and image size."""
        super(NaiveConvolutionalFeatureMap, self).__init__()
        self.convolution = ConvolutionalPlane(fsize, imsize)
        self.nonlinearity = TanhSigmoid(self.convolution.outsize)
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        return self.nonlinearity.fprop(self.convolution.fprop(inputs))
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        squash_inputs = self.convolution.fprop(inputs)
        squash_derivs = self.nonlinearity.bprop(dout, squash_inputs)
        return self.convolution.bprop(squash_derivs, inputs)
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        squash_inputs = self.convolution.fprop(inputs)
        squash_derivs = self.nonlinearity.bprop(dout, squash_inputs)
        return self.convolution.grad(squash_derivs, inputs)
    
    def initialize(self):
        """Initialize the module's weights."""
        self.convolution.initialize()
    
    def outsize(self):
        """Output size."""
        return self.convolution.outsize()
    
    def imsize(self):
        """Image input size."""
        return self.convolution.imsize()
    
    def fsize(self):
        """Filter shape."""
        return self.convolution.filter.shape

class ConvolutionalFeatureMap(ConvolutionalPlane):
    """
    A better implementation of a feature map with a convolution followed
    by a sigmoid (tanh) squashing function. 
    
    Uses a tied bias for every unit in the feature map.
    """
    def __init__(self, fsize, imsize, params=None, 
                 inner=TANH_INNER, outer=TANH_OUTER):
        """
        Initialize the feature map.
        
        * fsize -- a 2-tuple of odd values indicating the height and width
                   of the local neighbourhood to which each hidden unit is
                   connected.
        
        * imsize -- the height and width of the image that this plane will 
                    take as input.
        
        * inner/outer -- are values that the output are scaled by before
                         and after the squashing function is applied,
                         respectively.
        """
        super(ConvolutionalFeatureMap, self).__init__(fsize, imsize, params)
        self.inner = inner
        self.outer = outer
        self._outer_times_inner = outer * inner
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        super(ConvolutionalFeatureMap, self).fprop(inputs)
        out = self._out_array[self.active_slice]
        out *= self.inner
        np.tanh(out, out)
        out *= self.outer
        return out
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        derivs = self._squash_derivatives(dout, inputs)
        out = super(ConvolutionalFeatureMap, self).bprop(derivs, inputs)
        return out
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        derivs = self._squash_derivatives(dout, inputs)
        return super(ConvolutionalFeatureMap, self).grad(derivs, inputs)
    
        
    ########################### Private interface ###########################
    
    def _squash_derivatives(self, dout, inputs):
        """
        Do the first step in either grad or bprop, which is to get
        derivatives of the activations (convolved values) with respect to 
        the outputs of the nonlinearity.
        """
        # NOTE: This modifies the fprop array (self._out_array) to something
        # other than what might be expected after a call to fprop alone.
        
        # We want ConvolutionalPlane's fprop, not FeatureMap's fprop here
        squash_derivs = super(ConvolutionalFeatureMap, self).fprop(inputs) 
        squash_derivs *= self.inner
        np.cosh(squash_derivs, squash_derivs)
        squash_derivs **= -2.
        squash_derivs *= self._outer_times_inner
        squash_derivs *= dout
        return squash_derivs
    
class AveragePoolingFeatureMap(AveragePoolingPlane):
    """
    A feature map with one trainable weight and one trainable 
    bias, that downsamples its input at a given ratio by 
    averaging disjoint neighbourhoods, applies the weight 
    and bias and finally a sigmoid function.
    """
    def __init__(self, ratio, imsize, inner=TANH_INNER, outer=TANH_OUTER):
        super(AveragePoolingFeatureMap, self).__init__(ratio, imsize)
        self.params = np.empty((2,))
        self.weights = self.params[:1]
        self.biases = self.params[1:]
        self.inner = inner
        self.outer = outer
        self._outer_times_inner = outer * inner
        self._grad = np.empty(len(self.params))
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        outputs = super(AveragePoolingFeatureMap, self).fprop(inputs)
        outputs *= self.weights[0]
        outputs += self.biases[0]
        outputs *= self.inner
        np.tanh(outputs, outputs)
        outputs *= self.outer
        return outputs

    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        out = self._derivs_common(dout, inputs)
        out *= self.weights[0]
        return out
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        derivs = self._derivs_common(dout, inputs)
        self._grad[1:] = derivs.sum()
        self._grad[:1] = (derivs * inputs).sum()
        return self._grad
    
    def _derivs_common(self, dout, inputs):
        """Common code factored out of both bprop and grad."""
        derivs = self._squash_derivatives(dout, inputs)
        out = super(AveragePoolingFeatureMap, self).bprop(derivs, inputs)
        return out
        
    def _squash_derivatives(self, dout, inputs):
        """
        Do the first step in either grad or bprop, which is to get
        derivatives of the activations (convolved values) with respect to 
        the outputs of the nonlinearity.
        """
        # NOTE: This modifies the fprop array (self._out_array) to something
        # other than what might be expected after a call to fprop alone.
        
        # We want AveragePoolingPlane's fprop, not FeatureMap's fprop here
        squash_derivs = super(AveragePoolingFeatureMap, self).fprop(inputs)
        
        # The weight and bias from this layer.
        squash_derivs *= self.weights[0]
        squash_derivs += self.biases[0]
        
        # Actually evaluate f prime.
        squash_derivs *= self.inner
        np.cosh(squash_derivs, squash_derivs)
        squash_derivs **= -2.
        squash_derivs *= self._outer_times_inner
        squash_derivs *= dout
        return squash_derivs

    def initialize(self):
        """Initialize the parameters in this module."""
        fan_in = np.prod(self.ratio)
        std = fan_in**-0.5
        paramsize = 2
        self.params[:] = random.uniform(-2.4*std, 2.4*std, size=paramsize)
    
    
class MultiConvolutionalFeatureMap(TanhSigmoid):
    """
    A class that encapsulates a FeatureMap that has several distinct
    planes, each with their own set of convolutional weights, that are
    summed together and globally biased before being squashed by 
    a sigmoid nonlinearity.
    """
    def __init__(self, fsize, imsize, num, inner=TANH_INNER, 
                 outer=TANH_OUTER):
        """
        Construct a MultiConvolutionalFeatureMap.
        
        All parameters are as in ConvolutionalFeatureMap; num is the
        number of planes this MCFM has (and the length of the list
        that fprop, bprop and grad expect for the "inputs" argument).
        """
        # Filter size times the number of filters, plus a bias
        filter_elems = np.prod(fsize)
        self.params = np.empty(filter_elems * num + 1)
        self._grad = np.empty(len(self.params))
        self.planes = []
        assert num > 0
        for index in xrange(num):
            start = 1 + (filter_elems * index)
            stop = 1 + (filter_elems * (index + 1))
            thisparam = self.params[start:stop]
            thisgrad = self._grad[start:stop]
            thisplane = ConvolutionalPlane(fsize, imsize, 
                params=thisparam,
                grad=thisgrad,
                bias=False
            )
            self.planes.append(thisplane)
        outsize = thisplane.outsize
        self._out_array = np.empty(outsize)
        super(MultiConvolutionalFeatureMap, self).__init__(
            outsize,
            True,
            params=self.params,  # This is a bit of a kludge.
            grad=self._grad
        )
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        self._add_up(inputs)
        return super(MultiConvolutionalFeatureMap, self).fprop(self._out_array)
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        deriv = self._squash_derivatives(dout, inputs)
        derivs = [deriv] * len(self.planes)
        triples = izip(self.planes, derivs, inputs)
        return [plane.bprop(deriv, inp) for plane, deriv, inp in triples]
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        deriv, sig_inp = self._squash_derivatives(dout, inputs, True)
        
        # Since the gradient arrays for all of these planes are
        # subarrays of self._grad we don't actually need to return
        # this; just make the method calls.
        planegrads = [plane.grad(deriv, inp) for plane, inp in
                      izip(self.planes, inputs)]

        # Compute the bias gradient, which should be stored in
        # self._grad[0]
        super(MultiConvolutionalFeatureMap, self).grad(dout, sig_inp)
        return self._grad
    
    def initialize(self, multiplier=1):
        """Initialize the parameters in this module."""
        for plane in self.planes:
            plane.initialize(multiplier * len(self.planes), True)
        super(MultiConvolutionalFeatureMap, self).initialize()
    
    def _squash_derivatives(self, dout, inputs, return_sig_inp=False):
        """
        Do the first step in either grad or bprop, which is to get
        derivatives of the activations (convolved values) with respect to 
        the outputs of the nonlinearity.
        """
        sig_inp = self._add_up(inputs)
        deriv = super(MultiConvolutionalFeatureMap, self).bprop(dout, sig_inp)
        if return_sig_inp:
            return deriv, sig_inp
        else:
            return deriv
    
    def _add_up(self, inputs):
        """
        Add up the outputs of all the ConvolutionalPlanes and stick
        the result in self._out_array.
        """
        outputs = [pln.fprop(inp) for pln, inp in izip(self.planes, inputs)]
        out = self._out_array
        out[...] = 0.
        for convolved in outputs:
            self._out_array += convolved
        return self._out_array
    

