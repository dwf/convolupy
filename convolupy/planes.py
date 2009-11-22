"""
Module for 'planes': classes that perform convolution/subsampling, 
that serve as base classes for corresponding feature map classes.
"""

# Standard library imports
from itertools import izip

# NumPy/SciPy imports
import numpy as np
from numpy import random
from scipy import ndimage

# Local imports
from convolupy.base import BaseBPropComponent

class ConvolutionalPlane(BaseBPropComponent):
    """
    A learning module that implements a plane of a convolutional neural 
    network, without the squashing function -- that is, the input image 
    is convolved (correlated, really) with a discrete filter. This is
    equivalent in neural network parlance to a plane of hidden units with 
    tied weights replicated across the entire visual field which each
    receive input only from a certain local neighbourhood. The output of
    this component is the 'activations' of the hidden units, i.e. the input
    to the nonlinearity.
    
    Use the subclass ConvolutionalFeatureMap for a combination convolutional
    plane and elementwise sigmoid squashing function. This ought only to be
    used directly in higher layers when several ConvolutionalPlanes
    might feed the input of a single sigmoidal feature map.
    """
    def __init__(self, fsize, imsize, params=None, grad=None, 
                 bias=True, biastype='map'):
        """
        Initialize a convolutional plane for an image of size 'imsize'
        with convolutional filter of size fsize.
        
        'bias' determines whether to learn biases in this plane.
        
        'biastype' must be one of 'map' or 'unit'. 'map' fixes one bias
        parameter for the whole feature map; 'unit' allows an individual
        bias for every unit (it's unclear to me whether this would ever
        be a good idea).
        """
        odd = (num % 2 == 1 for num in fsize)
        if len(fsize) != 2 or not all(odd):
            raise ValueError('fsize must be length 2, both numbers odd')
        
        # Parameters for this layer and views onto them
        filter_elems = np.prod(fsize)
        outsize = self._output_from_im_and_filter_size(imsize, fsize)
        if bias:
            if biastype == 'map':
                bias_elems = 1
                bias_shape = (1,)
            elif biastype == 'unit':
                bias_elems = np.prod(outsize)
                bias_shape = outsize
            else:
                raise ValueError('biastype must be \'map\' or \'unit\'')
        else:
            bias_elems = 0
            bias_shape = 0

        super(ConvolutionalPlane, self).__init__(
            filter_elems + bias_elems, 
            params,
            grad
        )

        # Oversized output array so we can use convolve() on it
        self._out_array = np.empty(imsize)
        self._bprop_array = np.empty(imsize)
        
        # Views onto the filter and bias portion of the parameter vector
        self.filter = self.params[:filter_elems].reshape(fsize)
        self.biases = self.params[filter_elems:].reshape(bias_shape)
        
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        
        # Look ma, no copies!
        assert len(inputs.shape) == 2
        self._activate(inputs, self._out_array)
        activations = self._out_array[self.active_slice]
        return activations
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        assert inputs.shape == self._out_array.shape
        vsize, hsize = self._out_array[self.active_slice].shape
        out = self._bprop_array
        out[...] = 0.
        for row in xrange(self.filter.shape[0]):
            for col in xrange(self.filter.shape[1]):
                weight = self.filter[row, col]
                out[row:(row+vsize), col:(col+hsize)] += dout * weight
        return out
    
    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        vsize, hsize = self._out_array[self.active_slice].shape
        filter_elems = np.prod(self.filter.shape)
        grad_filter = self._grad[:filter_elems].reshape(self.filter.shape)
        grad_biases = self._grad[filter_elems:].reshape(self.biases.shape)
        
        for row in xrange(self.filter.shape[0]):
            for col in xrange(self.filter.shape[1]):
                cgrad = (dout * inputs[row:(row+vsize),
                    col:(col+hsize)]).sum()
                grad_filter[row, col] = cgrad
        if len(self.biases.shape) > 1:
            grad_biases[...] = dout 
        else:
            # This is simply a no-op when self.biases.shape = (0,)
            grad_biases[...] = dout.sum()
        return self._grad
    
    @property
    def outsize(self):
        """Output size."""
        imsize = self._out_array.shape
        fsize = self.filter.shape
        return self._output_from_im_and_filter_size(imsize, fsize)
    
    @property
    def fsize(self):
        """Filter shape."""
        return self.filter.shape
            
    @property
    def active_slice(self):
        """
        Active slice of the output array - that is, the slice
        containing the outputs of the convolution that are not NaN 
        because they are on the border.
        """
        
        voff, hoff = self._offsets
        
        # Image size
        imsize = self._out_array.shape
        
        # A slice-tuple representing the 'active' region of the output
        return (slice(voff, voff + imsize[0] - 2 * voff), 
            slice(hoff, hoff + imsize[1] - 2 * voff))
    
        
    ########################### Private interface ###########################
    
    @property
    def _offsets(self):
        """Vertical and horizontal offsets -- the padding around the input"""
        return self._offsets_from_filter_size(self.filter.shape)
    
    @staticmethod
    def _offsets_from_filter_size(fsize):
        """
        Given filter size, calculate the offsets at the borders of 
        the input image.
        """
        return [np.floor(dim / 2) for dim in fsize]
    
    @classmethod
    def _output_from_im_and_filter_size(cls, imsize, fsize):
        """Given image size and filter size, calculate size of the output."""
        offsets = cls._offsets_from_filter_size(fsize)
        return [size - 2 * off for off, size in izip(offsets, imsize)]
    
    def initialize(self, multiplier=1, always_add_bias=False):
        """
        Initialize the plane's weights.
        
        'multiplier' multiples the computed fan-in by a specified value.
        So if this is one of 5 sets of convolutional weights that are 
        summed and feed a single plane of sigmoids, multiplier=5 will 
        take this into account when initializing the weights.
        
        'always_add_bias' is a flag to add one to fan-in even if this
        module doesn't have a bias attached to it (i.e. if several of
        these things feed into one sigmoid with a bias on it)
        """
        fan_in = np.prod(self.filter.shape) # All filter weights
        fan_in *= multiplier
        # If this module adds biases, automatically add 1 to the fan-in
        if self.biases.size > 0 or always_add_bias:
            fan_in += 1
        std = fan_in**-0.5
        size = self.params.shape
        self.params[:] = random.uniform(low=-2.4*std, high=2.4*std, size=size)
    
    def _activate(self, inputs, out, cval=np.nan):
        """Generate input activities for neurons (convolve and add bias)."""
        out[...] = 0.
        ndimage.correlate(inputs, self.filter, mode='constant', cval=cval, 
            output=out)
        if self.biases.size > 0:
            out[self.active_slice] += self.biases

class AveragePoolingPlane(BaseBPropComponent):
    """
    A fixed module (no learnable parameters) that performs downsampling 
    by averaging in non-overlapping local neighbourhoods.
    """
    def __init__(self, ratio, imsize, *args, **kwargs):
        """
        Construct an AveragePoolingPlane that downsamples an image of size
        imsize at the given subsampling ratio.
        """
        super(AveragePoolingPlane, self).__init__(*args, **kwargs)
        if len(ratio) != 2 or len(imsize) != 2:
            raise ValueError('Both ratio and imsize must be length 2')
        elif any(dim_i % dim_r != 0 for dim_i, dim_r in izip(imsize, ratio)):
            raise ValueError('Image dimensions must be divisible by ratios')
        self.ratio = ratio
        size = [imdim / ratdim for imdim, ratdim in zip(imsize, ratio)]
        size += imsize[2:]
        self._out_array = np.empty(size)
        self._bprop_array = np.empty(imsize)
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        self._out_array[...] = 0.
        for row_start in xrange(self.ratio[0]):
            for col_start in xrange(self.ratio[1]):
                row_r = self.ratio[0]
                col_r = self.ratio[1]
                self._out_array += inputs[row_start::row_r, 
                    col_start::col_r, ...]
        self._out_array /= np.prod(self.ratio)
        return self._out_array
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        if inputs is not None:
            pass
        for rstart in xrange(self.ratio[0]):
            for cstart in xrange(self.ratio[1]):
                # Strides for row and column
                rstr = self.ratio[0]
                cstr = self.ratio[1]
                self._bprop_array[..., rstart::rstr, cstart::cstr] = dout
        self._bprop_array /= np.prod(self.ratio)
        return self._bprop_array

