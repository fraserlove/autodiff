import math
from autodiff import Value
from .utils import finite_grad, assert_gradients_close


class TestValueBasic:
    """Test basic Value operations."""

    def test_value_creation(self):
        """Test Value object creation."""
        v = Value(5.0)
        assert v.v == 5.0
        assert v.dv == 0
        assert v._prev == set()
        assert v._op == ''

    def test_value_with_label(self):
        """Test Value creation with label."""
        v = Value(3.0, label='test')
        assert v.v == 3.0
        assert v.label == 'test'

    def test_value_repr(self):
        """Test Value string representation."""
        v = Value(2.5)
        v.dv = 1.5
        assert repr(v) == 'Value(2.5, dv=1.5)'

    def test_value_equality(self):
        """Test Value equality (by reference)."""
        v1 = Value(5.0)
        v2 = Value(5.0)
        assert v1 is not v2  # Different objects
        assert v1.v == v2.v  # Same values


class TestValueArithmetic:
    """Test arithmetic operations on Value objects."""

    def test_addition(self):
        """Test addition operation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        
        assert c.v == 5.0
        assert c._op == '+'
        assert len(c._prev) == 2
        assert (a, 1) in c._prev
        assert (b, 1) in c._prev

    def test_addition_with_scalar(self):
        """Test addition with scalar values."""
        a = Value(2.0)
        c = a + 3.0
        
        assert c.v == 5.0
        assert c._op == '+'
        assert len(c._prev) == 2

    def test_right_addition(self):
        """Test right addition (scalar + Value)."""
        a = Value(2.0)
        c = 3.0 + a
        
        assert c.v == 5.0
        assert c._op == '+'

    def test_multiplication(self):
        """Test multiplication operation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        
        assert c.v == 6.0
        assert c._op == '*'
        assert len(c._prev) == 2
        assert (a, 3.0) in c._prev  # a * b, so gradient w.r.t. a is b
        assert (b, 2.0) in c._prev  # a * b, so gradient w.r.t. b is a

    def test_multiplication_with_scalar(self):
        """Test multiplication with scalar values."""
        a = Value(2.0)
        c = a * 3.0
        
        assert c.v == 6.0
        assert c._op == '*'

    def test_right_multiplication(self):
        """Test right multiplication (scalar * Value)."""
        a = Value(2.0)
        c = 3.0 * a
        
        assert c.v == 6.0
        assert c._op == '*'

    def test_power(self):
        """Test power operation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a ** b
        
        assert c.v == 8.0  # 2^3 = 8
        assert c._op == '**3.0'
        assert len(c._prev) == 1
        # Gradient of x^y w.r.t. x is y * x^(y-1) = 3 * 2^2 = 12
        assert (a, 12.0) in c._prev

    def test_power_with_scalar(self):
        """Test power with scalar values."""
        a = Value(2.0)
        c = a ** 3.0
        
        assert c.v == 8.0
        assert c._op == '**3.0'

    def test_negation(self):
        """Test negation operation."""
        a = Value(2.0)
        b = -a
        
        assert b.v == -2.0
        assert b._op == '*'

    def test_subtraction(self):
        """Test subtraction operation."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        
        assert c.v == 2.0
        assert c._op == '+'

    def test_right_subtraction(self):
        """Test right subtraction (scalar - Value)."""
        a = Value(2.0)
        c = 5.0 - a
        
        assert c.v == 3.0

    def test_division(self):
        """Test division operation."""
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        
        assert c.v == 3.0
        assert c._op == '*'

    def test_right_division(self):
        """Test right division (scalar / Value)."""
        a = Value(2.0)
        c = 6.0 / a
        
        assert c.v == 3.0


class TestValueMathematicalFunctions:
    """Test mathematical functions on Value objects."""

    def test_exp(self):
        """Test exponential function."""
        a = Value(1.0)
        b = a.exp()
        
        assert b.v == math.e
        assert b._op == 'exp'
        assert len(b._prev) == 1
        assert (a, math.e) in b._prev

    def test_log(self):
        """Test logarithm function."""
        a = Value(math.e)
        b = a.log()
        
        assert b.v == 1.0
        assert b._op == 'log'
        assert len(b._prev) == 1
        assert (a, 1/math.e) in b._prev

    def test_sin(self):
        """Test sine function."""
        a = Value(math.pi/2)
        b = a.sin()
        
        assert abs(b.v - 1.0) < 1e-10
        assert b._op == 'sin'
        assert len(b._prev) == 1
        # cos(π/2) ≈ 0, so gradient should be very small
        gradient = next(iter(b._prev))[1]
        assert abs(gradient) < 1e-10

    def test_cos(self):
        """Test cosine function."""
        a = Value(0.0)
        b = a.cos()
        
        assert b.v == 1.0
        assert b._op == 'cos'
        assert len(b._prev) == 1
        assert (a, 0.0) in b._prev  # -sin(0) = 0

    def test_tanh(self):
        """Test hyperbolic tangent function."""
        a = Value(0.0)
        b = a.tanh()
        
        assert b.v == 0.0
        assert b._op == 'tanh'
        assert len(b._prev) == 1
        assert (a, 1.0) in b._prev  # 1 - tanh(0)^2 = 1

    def test_relu(self):
        """Test ReLU function."""
        # Test positive input
        a = Value(2.0)
        b = a.relu()
        assert b.v == 2.0
        assert b._op == 'relu'
        assert len(b._prev) == 1
        assert (a, 1) in b._prev  # ReLU'(x) = 1 for x > 0
        
        # Test negative input
        a = Value(-2.0)
        b = a.relu()
        assert b.v == 0.0
        assert b._op == 'relu'
        assert len(b._prev) == 1
        assert (a, 0) in b._prev  # ReLU'(x) = 0 for x <= 0


class TestValueBackpropagation:
    """Test backpropagation through computation graphs."""

    def test_simple_backprop(self):
        """Test backpropagation on a simple expression."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        
        assert c.dv == 1.0  # Final node
        assert a.dv == 3.0  # d/da(a*b) = b
        assert b.dv == 2.0  # d/db(a*b) = a

    def test_chain_rule(self):
        """Test backpropagation with chain rule."""
        a = Value(2.0)
        b = a * a  # b = a^2
        c = b * a  # c = a^3
        c.backward()
        
        assert c.dv == 1.0
        # d/da(a^3) = 3*a^2 = 3*4 = 12
        assert a.dv == 12.0

    def test_complex_expression(self):
        """Test backpropagation on a complex expression."""
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        
        # f = (a + b) * c
        d = a + b
        e = d * c
        e.backward()
        
        assert e.dv == 1.0
        assert d.dv == 4.0  # d/de(d*c) = c
        assert c.dv == 5.0  # d/de(d*c) = d = a + b = 5
        assert a.dv == 4.0  # d/dd(a+b) = 1, so d/da = 1 * 4 = 4
        assert b.dv == 4.0  # d/dd(a+b) = 1, so d/db = 1 * 4 = 4

    def test_multiple_paths(self):
        """Test backpropagation with multiple paths to same variable."""
        a = Value(2.0)
        b = a * a  # a^2
        c = a * a  # a^2 again
        d = b + c  # 2*a^2
        d.backward()
        
        assert d.dv == 1.0
        # d/da(2*a^2) = 4*a = 8
        assert a.dv == 8.0

    def test_zero_gradients(self):
        """Test that gradients are properly accumulated."""
        a = Value(2.0)
        b = a * a
        c = a * a
        d = b + c
        d.backward()
        
        # a should have gradient from both b and c
        assert a.dv == 8.0  # 2*a + 2*a = 4*a = 8


class TestValueGradientChecking:
    """Test gradient computation using finite differences."""

    def test_quadratic_gradient(self):
        """Test gradient of quadratic function."""
        def f(x):
            return x * x
        
        x = Value(2.0)
        y = f(x)
        y.backward()
        
        # Finite difference gradient
        finite_grad_val = finite_grad(f, x)
        
        assert_gradients_close(x.dv, finite_grad_val)

    def test_cubic_gradient(self):
        """Test gradient of cubic function."""
        def f(x):
            return x * x * x
        
        x = Value(1.5)
        y = f(x)
        y.backward()
        
        # Finite difference gradient
        finite_grad_val = finite_grad(f, x)
        
        assert_gradients_close(x.dv, finite_grad_val)

    def test_exponential_gradient(self):
        """Test gradient of exponential function."""
        def f(x):
            return x.exp()
        
        x = Value(1.0)
        y = f(x)
        y.backward()
        
        # Finite difference gradient
        finite_grad_val = finite_grad(f, x)
        
        assert_gradients_close(x.dv, finite_grad_val)

    def test_sine_gradient(self):
        """Test gradient of sine function."""
        def f(x):
            return x.sin()
        
        x = Value(1.0)
        y = f(x)
        y.backward()
        
        # Finite difference gradient
        finite_grad_val = finite_grad(f, x)
        
        assert_gradients_close(x.dv, finite_grad_val)

    def test_complex_function_gradient(self):
        """Test gradient of complex function."""
        def f(x):
            return (x * x).exp()
        
        x = Value(0.5)
        y = f(x)
        y.backward()
        
        # Finite difference gradient
        finite_grad_val = finite_grad(f, x)
        
        assert_gradients_close(x.dv, finite_grad_val)
