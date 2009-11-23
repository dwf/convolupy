"""
Useful numerical constants. Right now just a home for the inner
and outer tanh scaling factors.
"""

from __future__ import division

# Chosen so that the sigmoid has fixed points at -1 and +1, which are also
# the points of maximum second derivative. This aids in fast convergence. 
TANH_OUTER = 1.7159
TANH_INNER = 2 / 3

