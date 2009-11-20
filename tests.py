from unittest import TestCase

import numpy as np
from numpy import random
from numpy.testing.utils  import assert_array_almost_equal

from nose.tools import raises

from convolupy.planes import ConvolutionalPlane, AveragePoolingPlane
from convolupy.maps import NaiveConvolutionalFeatureMap
from convolupy.maps import ConvolutionalFeatureMap, AveragePoolingFeatureMap
from convolupy.maps import MultiConvolutionalFeatureMap
from convolupy.sigmoids import TanhSigmoid


def fd_grad(func, x_in, tol=1.0e-5):
    """
    Approximates the gradient of f with finite differences, moving
    half of tol in either direction on each axis.
    """
    num = len(x_in)
    grad = np.zeros(num)
    for i in xrange(num):
        aaa = x_in.copy()
        bbb = x_in.copy()
        aaa[i] = aaa[i] - tol / 2.0
        bbb[i] = bbb[i] + tol / 2.0
        grad[i] = (func(bbb) - func(aaa))/(bbb[i] - aaa[i])
    return grad

def summed_objective_func(inputs, module):
    """Function that sums the fprop output of a module."""
    inputs = inputs.reshape(module.imsize)
    return module.fprop(inputs).sum()

def summed_objective_gradient(inputs, module):
    """
    Gradient wrt input of a function that sums the fprop output of a
    module.
    """
    inputs = inputs.reshape(module.imsize)
    return module.bprop(np.ones(module.outsize), inputs)

def summed_objective_params_func(params, inputs, module):
    """
    Function that sets parameters of a module and sums the fprop output
    of the module.
    """
    module.params[:] = params
    return module.fprop(inputs.reshape(module.imsize)).sum()

def summed_objective_params_gradient(params, inputs, module):
    """
    Gradient wrt parameters of the module on a given input.
    """
    module.params[:] = params
    return module.grad(np.ones(module.outsize), inputs)

def check_input_gradient(module, inputs):
    """
    Given a module, and inputs, checks the gradient with respect 
    to the input.
    """
    n_elems = np.prod(module.imsize)
    inputs = inputs.reshape(np.prod(inputs.shape))
    func = lambda inputs: summed_objective_func(inputs, module)
    approx_grad = fd_grad(func, inputs)
    real_grad = summed_objective_gradient(inputs, module).reshape(n_elems)
    assert_array_almost_equal(real_grad, approx_grad)

def check_parameter_gradient(module, inputs, params):
    """
    Given a module, an objective function (one of the ones 
    specified in this Python module) and inputs/parameters, checks the
    gradient with respect to the parameters.
    """
    func = lambda params: summed_objective_params_func(params, inputs, module)
    approx_grad = fd_grad(func, params)
    real_grad = summed_objective_params_gradient(params, inputs, module)
    assert_array_almost_equal(real_grad, approx_grad)

def test_input_gradients_basic():
    """Test the input gradient."""
    module_classes = [
        ConvolutionalPlane, 
        ConvolutionalFeatureMap,
        AveragePoolingPlane, 
        AveragePoolingFeatureMap
    ]
    for module_class in module_classes:
        module = module_class((5, 5), (20, 20))
        module.initialize()
        inputs = random.normal(size=module.imsize)
        yield check_input_gradient, module, inputs

def test_parameter_gradients_basic():
    """Test the parameter gradient."""
    module_classes = [
        ConvolutionalPlane, 
        ConvolutionalFeatureMap,
        AveragePoolingFeatureMap,
        TanhSigmoid
    ]
    for module_class in module_classes:
        if module_class is TanhSigmoid:
            module = module_class((20, 20), bias=True)
        else:
            module = module_class((5, 5), (20, 20))
        module.initialize()
        inputs = random.normal(size=module.imsize)
        params = random.normal(size=module.params.shape)
        yield check_parameter_gradient, module, inputs, params

def test_convolutional_plane_params_gradient_unit_bias():
    module = ConvolutionalPlane((5, 5), (20, 20), biastype='unit')
    module.initialize()
    inputs = random.normal(size=(20, 20))
    params = random.normal(size=len(module.params))
    check_parameter_gradient(module, inputs, params)

def test_convolutional_plane_params_gradient_no_bias():
    module = ConvolutionalPlane((5, 5), (20, 20), bias=False)
    module.initialize()
    inputs = random.normal(size=(20, 20))
    params = random.normal(size=len(module.params))
    check_parameter_gradient(module, inputs, params)

def check_supplied_params_vector_is_assigned(plen, moduleclass, args, kwargs):
    params = np.empty(plen)
    module = moduleclass(params=params, *args, **kwargs)
    assert params is module.params

def check_supplied_params_and_grad_assigned(plen, moduleclass, args, kwargs):
    params = np.empty(plen)
    grad = np.empty(plen)
    module = moduleclass(params=params, grad=grad, *args, **kwargs)
    dout = random.normal(size=module.outsize)
    inputs = random.normal(size=module.imsize)
    assert params is module.params
    assert module.grad(dout, inputs) is grad

def test_supplied_params_vectors():
    tests = [
        (
            26, # parameter length required
            ConvolutionalPlane, # class
            ((5, 5), (20, 20)), # args to constructor
            {} # kwargs to constructor
        ),
        (
            25, # parameter length required
            ConvolutionalPlane, # class
            ((5, 5), (20, 20)), # args to constructor
            dict(bias=False)    # kwargs to constructor
        ),
        (
            25 + 16 * 16, # parameter length required
            ConvolutionalPlane, # class
            ((5, 5), (20, 20)), # args to constructor
            dict(biastype='unit') # kwargs to constructor
        ),
        (
            1, # parameter length required
            TanhSigmoid, # class
            ((20, 20), True), # args to constructor
            {} # kwargs to constructor
        )
    ]
    for this_test in tests:
        yield (check_supplied_params_vector_is_assigned,) + this_test
        yield (check_supplied_params_and_grad_assigned,) + this_test

def test_convolutional_feature_map_agrees_with_naive_version_fprop():
    cmap = ConvolutionalFeatureMap((5, 5), (20, 20))
    cmap.initialize()
    cmap_naive = NaiveConvolutionalFeatureMap((5, 5), (20, 20))
    
    # Sync up their parameters
    cmap_naive.convolution.params[:] = cmap.params
        
    inputs = random.normal(size=cmap.imsize)
    assert_array_almost_equal(cmap.fprop(inputs), cmap_naive.fprop(inputs))

def test_convolutional_feature_map_agrees_with_naive_version_grad():
    cmap = ConvolutionalFeatureMap((5, 5), (20, 20))
    cmap.initialize()
    cmap_naive = NaiveConvolutionalFeatureMap((5, 5), (20, 20))
    cmap_naive.initialize()
    
    # Sync up their parameters
    cmap_naive.convolution.params[:] = cmap.params
    inputs = random.normal(size=cmap.imsize)
    dout = random.normal(size=cmap.outsize)
    assert_array_almost_equal(cmap.grad(dout, inputs), 
                              cmap_naive.grad(dout, inputs))

def test_convolutional_feature_map_agrees_with_naive_version_bprop():
    cmap = ConvolutionalFeatureMap((5, 5), (20, 20))
    cmap.initialize()
    cmap_naive = NaiveConvolutionalFeatureMap((5, 5), (20, 20))
    
    # Sync up their parameters
    cmap_naive.convolution.params[:] = cmap.params
    
    inputs = random.normal(size=cmap.imsize)
    dout = random.normal(size=cmap.outsize)
    assert_array_almost_equal(cmap.bprop(dout, inputs), 
                              cmap_naive.bprop(dout, inputs))


def test_average_pooling_plane_fprop():
    """
    Test that AveragePoolingPlane really does what we think it does
    on a forward pass.
    """
    applane = AveragePoolingPlane((2, 2), (4, 4))
    inp = np.arange(1,17).reshape((4,4))
    out = applane.fprop(inp)
    assert_array_almost_equal(out, 
        np.array([
            [(1+2+5+6)/4., (3+4+7+8)/4.],
            [(9+10+13+14)/4., (11+12+15+16)/4.]
        ])
    )

def test_average_pooling_feature_map_fprop():
    """
    Test that AveragePoolingFeatureMap really does what we think it does
    on a forward pass.
    """
    apfmap = AveragePoolingFeatureMap((2, 2), (4, 4))
    apfmap.biases[:] = 0.
    apfmap.weights[:] = 1.
    inp = np.arange(1,17).reshape((4,4))
    out = apfmap.fprop(inp)
    assert_array_almost_equal(out, 
        1.7159 * np.tanh( 2./3. * np.array([
            [(1+2+5+6)/4., (3+4+7+8)/4.],
            [(9+10+13+14)/4., (11+12+15+16)/4.]
        ])
    ))
def test_average_pooling_feature_map_fprop_bias():
    """
    Test that AveragePoolingFeatureMap really does what we think it does
    on a forward pass.
    """
    apfmap = AveragePoolingFeatureMap((2, 2), (4, 4))
    apfmap.biases[:] = -9.
    apfmap.weights[:] = 1.
    inp = np.arange(1,17).reshape((4,4))
    out = apfmap.fprop(inp)
    assert_array_almost_equal(out,
        1.7159 * np.tanh( 2./3. * (-9 + 
        np.array([
            [(1+2+5+6)/4., (3+4+7+8)/4.],
            [(9+10+13+14)/4., (11+12+15+16)/4.]
        ])
    )))

def test_average_pooling_feature_map_fprop_weight():
    """
    Test that AveragePoolingFeatureMap really does what we think it does
    on a forward pass.
    """
    apfmap = AveragePoolingFeatureMap((2, 2), (4, 4))
    apfmap.biases[:] =  0.
    apfmap.weights[:] = 4.4
    inp = np.arange(1,17).reshape((4,4))
    out = apfmap.fprop(inp)
    assert_array_almost_equal(out,
        1.7159 * np.tanh( 2./3. * 4.4 * 
        np.array([
            [(1+2+5+6)/4., (3+4+7+8)/4.],
            [(9+10+13+14)/4., (11+12+15+16)/4.]
        ])
    ))


def test_average_pooling_feature_map_fprop_weight_and_bias():
    """
    Test that AveragePoolingFeatureMap really does what we think it does
    on a forward pass.
    """
    apfmap = AveragePoolingFeatureMap((2, 2), (4, 4))
    apfmap.biases[:] =  -9.
    apfmap.weights[:] = 4.4
    inp = np.arange(1,17).reshape((4,4))
    out = apfmap.fprop(inp)
    assert_array_almost_equal(out,
        1.7159 * np.tanh( 2./3. * (-9 + 4.4 * 
        np.array([
            [(1+2+5+6)/4., (3+4+7+8)/4.],
            [(9+10+13+14)/4., (11+12+15+16)/4.]
        ])
    )))


def test_multi_convolutional_feature_map_fprop():
    cplane1 = ConvolutionalPlane((5, 5), (20, 20), bias=False)
    cplane2 = ConvolutionalPlane((5, 5), (20, 20), bias=False)
    sigmoid = TanhSigmoid((16, 16), bias=True)
    mfmap = MultiConvolutionalFeatureMap((5, 5), (20,20), 2)
    mfmap.initialize()
    cplane1.params[:] = mfmap.planes[0].params
    cplane2.params[:] = mfmap.planes[1].params
    sigmoid.params[:] = mfmap.params
    inputs1 = random.normal(size=(20, 20))
    inputs2 = random.normal(size=(20, 20))
    control = sigmoid.fprop(cplane1.fprop(inputs1) + cplane2.fprop(inputs2))
    mfmap_out = mfmap.fprop([inputs1, inputs2])
    assert_array_almost_equal(control, mfmap_out)

def test_multi_convolutional_feature_map_singleplane_bprop():
    size = (20, 20)
    elems = np.prod(size)
    fsize = (5, 5)
    osize = (16, 16)
    mfmap = MultiConvolutionalFeatureMap(fsize, size, 1)
    mfmap.initialize()
    in1 = random.normal(size=size)
    dout = np.ones(osize)
    bprop = lambda inp: mfmap.bprop(dout, inp)
    grad1 = lambda var: bprop((var.reshape(size),))[0].reshape(elems)
    func1 = lambda var: mfmap.fprop((var.reshape(size),)).sum()
    varied_input = random.normal(size=size)
    fd_grad1 = fd_grad(func1, varied_input.reshape(elems), 1e-4)
    real_grad1 = grad1(varied_input)
    assert_array_almost_equal(fd_grad1, real_grad1)

def test_multi_convolutional_feature_map_twoplane_bprop():
    size = (20, 20)
    elems = np.prod(size)
    fsize = (5, 5)
    osize = (16, 16)
    mfmap = MultiConvolutionalFeatureMap(fsize, size, 2)
    mfmap.initialize()
    
    inp = random.normal(size=size)
    cnst = random.normal(size=size)

    dout = np.ones(osize)

    fprop = lambda y: mfmap.fprop((y.reshape(size), cnst)).sum()
    approximate = fd_grad(fprop, inp.reshape(400))
    actual = mfmap.bprop(np.ones(osize), (inp, cnst))[0].reshape(400)
    assert_array_almost_equal(approximate, actual)

    # Swap the order - should make no difference
    fprop = lambda y: mfmap.fprop((cnst, y.reshape(size))).sum()
    approximate = fd_grad(fprop, inp.reshape(400))
    actual = mfmap.bprop(np.ones(osize), (cnst, inp))[1].reshape(400)
    assert_array_almost_equal(approximate, actual)


class ConvolutionalPlaneExceptionsTester(TestCase):
    def setUp(self):
        self._moduleclass = ConvolutionalPlane
    
    @raises(ValueError)
    def test_too_many_filter_dims(self):
        foo = self._moduleclass((5, 5, 7), (6, 6))
    
    @raises(ValueError)
    def test_too_few_filter_dims(self):
        foo = self._moduleclass((5,), (6, 6))
    
    @raises(ValueError)
    def test_not_odd_filter_dims(self):
        foo = self._moduleclass((5, 4), (6, 6))
    
    @raises(ValueError)
    def test_not_odd_filter_dims(self):
        foo = self._moduleclass((5, 4), (6, 6))
    
    @raises(ValueError)
    def test_non_odd_filter_dims(self):
        foo = self._moduleclass((4, 5), (6, 6))
    
    @raises(ValueError)
    def test_non_odd_filter_dims_other(self):
        foo = self._moduleclass((4, 5), (6, 6))
    
    @raises(ValueError)
    def test_bad_bias_type(self):
        if self._moduleclass is ConvolutionalPlane:
            foo = self._moduleclass((5, 5), (6, 6), biastype='foo')
        else:
            # We don't want to test subclasses since they don't inherit this
            # behaviour
            raise ValueError()
    
    @raises(ValueError)
    def test_bad_params_scalar(self):
        foo = self._moduleclass((5, 5), (6, 6), params=5)
    
    @raises(ValueError)
    def test_bad_params_rank2(self):
        foo = self._moduleclass((5, 5), (6, 6), params=np.array([[4,5]]))
    
    @raises(ValueError)
    def test_bad_params(self):
        foo = self._moduleclass((5, 5), (6, 6), params=np.array([[4,5]]))
    
    @raises(ValueError)
    def test_bad_params(self):
        foo = self._moduleclass((5, 5), (6, 6), params=np.zeros(25))
    

class ConvolutionalFMapExceptionsTester(ConvolutionalPlaneExceptionsTester):
    def setUp(self):
        self._moduleclass = ConvolutionalFeatureMap
    

class TanhSigmoidExceptionsTester(TestCase):
    @raises(ValueError)
    def test_sigmoid_initialize_raises_if_no_parameters(self):
        foo = TanhSigmoid((5,5))
        foo.initialize()

class AveragePoolingPlaneExceptionsTester(TestCase):
    @raises(ValueError)
    def test_ratios_length_less_than_2(self):
        foo = AveragePoolingPlane((5,), (25, 25))
    
    @raises(ValueError)
    def test_ratios_length_greater_than_2(self):
        foo = AveragePoolingPlane((5, 5, 5), (25, 25))
    
    @raises(ValueError)
    def test_ratio_nondivisible(self):
        foo = AveragePoolingPlane((5, 5), (24, 24))
    
