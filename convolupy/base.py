"""Base class for backpropagation components."""

class BaseBPropComponent(object):
    """Base class for all back-propagatable components."""
    def __init__(self):
        """Dummy constructor. This class should never be instantiated."""
        # To satisfy pylint
        self._out_array = None
        self._bprop_array = None
    
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
    

