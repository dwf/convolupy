"""
Layers are collections of feature maps or planes with multiple output
to the next layer.
"""
from itertools import izip

import numpy as np

from convolupy.base import BaseBPropComponent
from convolupy.maps import ConvolutionalFeatureMap, AveragePoolingFeatureMap, \
        MultiConvolutionalFeatureMap

class AbstractFeatureMapLayer(BaseBPropComponent):
    """Common methods for a bunch of layer classes."""
    def __init__(self, *args, **kwargs):
        super(AbstractFeatureMapLayer, self).__init__(*args, **kwargs)
        self.maps = []

    def _create_maps(self, mapclass, params_per, num, *args, **kwargs):
        """Create feature maps of a given class in self.maps."""
        # Don't let params and grad interfere
        if 'params' in kwargs:
            del kwargs['params']
        if 'grad' in kwargs:
            del kwargs['grad']
        
        for index in xrange(num):
            params_range = slice(params_per * index, params_per * (index + 1))
            thismap = mapclass(
                params=self.params[params_range],
                grad=self._grad[params_range],
                *args,
                **kwargs
            )
            self.maps.append(thismap)

    def initialize(self, *args, **kwargs):
        """Initialize the module's weights."""
        for fmap in self.maps:
            fmap.initialize(*args, **kwargs)

    def fprop(self, inputs):
        """Forward propagate input through this module."""
        assert len(inputs) == len(self.maps)
        return [fmap.fprop(inp) for fmap, inp in izip(self.maps, inputs)]

    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        return self._common_bprop_grad('bprop', dout, inputs)

    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        return self._common_bprop_grad('grad', dout, inputs)

    def _common_bprop_grad(self, meth, dout, inputs):
        """
        Common code factored out of bprop and grad. 'meth' is the 
        name of the method to call on the substituent objects (either
        'bprop' or 'grad').
        """
        assert len(dout) == len(self.maps) and len(inputs) == len(self.maps)
        return [
            getattr(fmap, meth)(deriv, inp)
            for fmap, deriv, inp in izip(self.maps, dout, inputs)
        ]

class ConvolutionalFeatureMapLayer(AbstractFeatureMapLayer):
    """A layer of ConvolutionalFeatureMaps."""
    def __init__(self, fsize, imsize, num, **kwargs):
        params_per = np.prod(fsize) + 1
        super(ConvolutionalFeatureMapLayer, self).__init__(
            nparams=params_per * num,
            **kwargs
        )
        self._create_maps(ConvolutionalFeatureMap,
                          params_per,
                          num,
                          fsize,
                          imsize,
                          **kwargs)


class AveragePoolingFeatureMapLayer(AbstractFeatureMapLayer):
    """A layer of AveragePoolingFeatureMaps."""
    def __init__(self, ratio, imsize, num, **kwargs):
        params_per = 2
        super(AveragePoolingFeatureMapLayer, self).__init__(
            nparams=params_per * num,
            **kwargs
        )
        self._create_maps(AveragePoolingFeatureMap,
                          params_per,
                          num,
                          ratio,
                          imsize,
                          **kwargs)


class MultiConvolutionalFeatureMapLayer(AbstractFeatureMapLayer):
    """A layer of MultiConvolutionalFeatureMaps."""
    def __init__(self, fsize, imsize, nummaps, connections, **kwargs):
        numparams = np.array([np.prod(fsize) * len(conn) + 1
                              for conn in connections])
        super(MultiConvolutionalFeatureMapLayer, self).__init__(
            nparams=np.sum(numparams),
            **kwargs
        )
        # Make sure the connections list is valid 
        assert len(connections) == len(nummaps)
        assert all(all(index < nummaps for index in conn)
                   for conn in connections)
        
        # Figure out which slices of the params and grad array to use for each
        upper = np.cumsum(numparams)
        lower = upper - numparams
        slices = [slice(start, stop) for start, stop in izip(lower, upper)]
        self.maps = []
        self.connections = connections

        for index in range(nummaps):
            thismap = MultiConvolutionalFeatureMap(
                fsize, 
                imsize,
                upper[index] - lower[index],
                params=self.params[slices[index]],
                grad=self._grad[slices[index]]
            )
            self.maps.append(thismap)

    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        out = []
        for index, fmap in enumerate(self.maps):
            theseinputs = [inputs[number] for number in self.connections[index]]
            out.append(fmap.fprop(theseinputs))
        return out

    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        # Oh damn this will be tricky.
        pass

    def grad(self, dout, inputs):
        """
        Gradient of the error with respect to the parameters of this module.
        
        Parameters:
            * dout -- derivative of the outputs of this module
                (will be size of input - size of filter + 1, elementwise)
            * inputs -- inputs to this module
        """
        out = []
        for index, fmap in enumerate(self.maps):
            theseinputs = [inputs[number] for number in self.connections[index]]
            out.append(fmap.grad(dout, theseinputs))
        return out
