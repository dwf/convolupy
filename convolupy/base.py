"""Base class for backpropagation components."""

import numpy as np

class BaseBPropComponent(object):
    """Base class for all back-propagatable components."""
    def __init__(self, nparams=None, params=None, grad=None):
        """Dummy constructor. This class should never be instantiated."""
        # To satisfy pylint
        self._out_array = None
        self._bprop_array = None
        
        if params is None:
            if nparams is not None:
                self.params = np.empty((nparams,))
            else:
                self.params = None
        else:
            if not hasattr(params, 'shape') or params.ndim != 1:
                raise ValueError('params must be rank 1 array if supplied')
            elif params.size < nparams:
                raise ValueError('params smaller than required (%d)' % nparams)
            self.params = params

        if grad is None:
            if nparams is not None:
                self._grad = np.empty((nparams,))
            else:
                self._grad = None
        else:
            if not hasattr(grad, 'shape') or grad.ndim != 1:
                raise ValueError('grad must be rank 1 array if supplied')
            elif grad.size < nparams:
                raise ValueError('grad smaller than required (%d)' % nparams)
            self._grad = grad
    
    @property
    def outsize(self):
        """Output size."""
        return self._out_array.shape
    
    @property
    def imsize(self):
        """Image input size."""
        return self._bprop_array.shape
    
    def initialize(self, *args, **kwargs):
        """Initialize the parameters in this module, if any."""
        pass
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        ishp = 'x'.join(str(x) for x in inputs.shape)
        raise NotImplementedError('fprop(input@%s): %s' % (ishp, self))
    
    def bprop(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        dshp = 'x'.join(str(x) for x in dout.shape)
        ishp = 'x'.join(str(x) for x in inputs.shape)
        raise NotImplementedError(
            'bprop(dout@%s, input@%s): %s' % (dshp, ishp, str(self))
        )
    
    def grad(self, dout, inputs):
        """
        Backpropagate derivatives through this module to get derivatives
        with respect to this module's input.
        """
        dshp = 'x'.join(str(x) for x in dout.shape)
        ishp = 'x'.join(str(x) for x in inputs.shape)
        raise NotImplementedError(
            'grad(dout@%s, input@%s): %s' % (dshp, ishp, str(self))
        )
    
    def __str__(self):
        """Return a sensible string representation of a bprop module."""
        if hasattr(self, 'ratio'):
            ratio = getattr(self, 'ratio')
            aux = " (downsampling @ %s)" % 'x'.join(str(x) for x in ratio)
        elif hasattr(self, 'filter'):
            fsize = getattr(self, 'fsize')
            aux = " (filtering @ %s)" % 'x'.join(str(x) for x in fsize)
        else:
            aux = ""
        name = self.__class__.__name__
        resolution = 'x'.join(str(x) for x in self.imsize)
        return "%s instance @ %s" % (name, resolution) + aux
    

