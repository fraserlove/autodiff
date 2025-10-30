import numpy as np
from autodiff import Value


def finite_grad(f, x, eps=1e-8):
    """
    Compute the finite difference gradient of function f at point x.
    
    Args:
        f: Function that takes a Value and returns a Value
        x: Value object to compute gradient for
        eps: Step size for finite differences (default: 1e-8)
    
    Returns:
        Value object containing the finite difference gradient
    """
    return (f(Value(x.v + eps)) - f(Value(x.v - eps))) / (2 * eps)


def assert_gradients_close(computed_grad, finite_grad, atol=1e-6, rtol=1e-6):
    """
    Assert that computed and finite difference gradients are close.
    
    Args:
        computed_grad: Computed gradient (Value object or numeric)
        finite_grad: Finite difference gradient (Value object or numeric)
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    if hasattr(computed_grad, 'v'):
        computed_val = computed_grad.v
    else:
        computed_val = computed_grad
    
    if hasattr(finite_grad, 'v'):
        finite_val = finite_grad.v
    else:
        finite_val = finite_grad
    
    assert np.isclose(computed_val, finite_val, atol=atol, rtol=rtol), \
        f"Gradients don't match: computed={computed_val}, finite={finite_val}"
