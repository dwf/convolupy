from itertools import izip

import numpy as np

from convolupy.base import BaseBPropComponent

class Network(BaseBPropComponent):
    
    def fprop(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.fprop(current)
        return current

    def bprop(self, dout, inputs, chain=None):
        if chain is None:
            current = inputs
            for layer in self.layers:
                chain.append(current)
                current = layer.fprop(current)
        current_dout = dout
        for layer, current_input in izip(self.layers[::-1], chain[::-1]):
            current_dout = layer.bprop(current_dout, current_input)
        return current_dout

    def grad(self, dout, inputs, chain=None):
        if chain is None:
            curr
