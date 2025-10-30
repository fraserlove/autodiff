import numpy as np
from autodiff import Value, Neuron, Layer, MLP, Module
from .utils import finite_grad, assert_gradients_close


class TestModule:
    """Test the base Module class."""

    def test_module_creation(self):
        """Test Module creation."""
        module = Module()
        assert isinstance(module, Module)
        assert module.parameters() == []

    def test_zero_grad(self):
        """Test zero_grad on empty module."""
        module = Module()
        module.zero_grad()  # Should not raise error


class TestNeuron:
    """Test the Neuron class."""

    def test_neuron_creation(self):
        """Test Neuron creation."""
        neuron = Neuron(3)
        assert len(neuron.weights) == 3
        assert isinstance(neuron.bias, Value)
        assert len(neuron.parameters()) == 4  # 3 weights + 1 bias

    def test_neuron_parameters(self):
        """Test that parameters are Value objects."""
        neuron = Neuron(2)
        params = neuron.parameters()
        
        assert len(params) == 3
        assert all(isinstance(p, Value) for p in params)
        assert neuron.bias in params
        assert all(w in params for w in neuron.weights)

    def test_neuron_forward_pass(self):
        """Test neuron forward pass."""
        # Create neuron with known weights
        neuron = Neuron(2)
        neuron.weights[0].v = 1.0
        neuron.weights[1].v = 2.0
        neuron.bias.v = 0.5
        
        x = [1.0, 2.0]
        output = neuron(x)
        
        # Expected: tanh(1*1 + 2*2 + 0.5) = tanh(5.5)
        expected = np.tanh(5.5)
        assert abs(output.v - expected) < 1e-10
        assert isinstance(output, Value)

    def test_neuron_gradient_flow(self):
        """Test that gradients flow through neuron."""
        neuron = Neuron(2)
        x = [1.0, 2.0]
        output = neuron(x)
        output.backward()
        
        # All parameters should have gradients
        for param in neuron.parameters():
            assert param.dv != 0

    def test_neuron_repr(self):
        """Test neuron string representation."""
        neuron = Neuron(3)
        assert repr(neuron) == 'Neuron(3)'

    def test_neuron_zero_grad(self):
        """Test zero_grad on neuron."""
        neuron = Neuron(2)
        x = [1.0, 2.0]
        output = neuron(x)
        output.backward()
        
        # Check gradients are non-zero
        assert any(p.dv != 0 for p in neuron.parameters())
        
        # Zero gradients
        neuron.zero_grad()
        assert all(p.dv == 0 for p in neuron.parameters())


class TestLayer:
    """Test the Layer class."""

    def test_layer_creation(self):
        """Test Layer creation."""
        layer = Layer(2, 3)
        assert len(layer.neurons) == 3
        assert all(isinstance(neuron, Neuron) for neuron in layer.neurons)
        assert all(len(neuron.weights) == 2 for neuron in layer.neurons)

    def test_layer_parameters(self):
        """Test layer parameters."""
        layer = Layer(2, 3)
        params = layer.parameters()
        
        # Each neuron has 3 parameters (2 weights + 1 bias)
        # 3 neurons * 3 parameters = 9 total
        assert len(params) == 9
        assert all(isinstance(p, Value) for p in params)

    def test_layer_forward_pass_single_output(self):
        """Test layer forward pass with single output."""
        layer = Layer(2, 1)
        x = [1.0, 2.0]
        output = layer(x)
        
        assert isinstance(output, Value)
        # Should be the output of the single neuron

    def test_layer_forward_pass_multiple_outputs(self):
        """Test layer forward pass with multiple outputs."""
        layer = Layer(2, 3)
        x = [1.0, 2.0]
        outputs = layer(x)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert all(isinstance(o, Value) for o in outputs)

    def test_layer_gradient_flow(self):
        """Test that gradients flow through layer."""
        layer = Layer(2, 2)
        x = [1.0, 2.0]
        outputs = layer(x)
        
        # Sum outputs to create a scalar loss
        loss = sum(outputs)
        loss.backward()
        
        # All parameters should have gradients
        for param in layer.parameters():
            assert param.dv != 0

    def test_layer_repr(self):
        """Test layer string representation."""
        layer = Layer(2, 3)
        repr_str = repr(layer)
        assert 'Layer of' in repr_str
        assert 'Neuron(2)' in repr_str

    def test_layer_zero_grad(self):
        """Test zero_grad on layer."""
        layer = Layer(2, 2)
        x = [1.0, 2.0]
        outputs = layer(x)
        loss = sum(outputs)
        loss.backward()
        
        # Check gradients are non-zero
        assert any(p.dv != 0 for p in layer.parameters())
        
        # Zero gradients
        layer.zero_grad()
        assert all(p.dv == 0 for p in layer.parameters())


class TestMLP:
    """Test the MLP class."""

    def test_mlp_creation(self):
        """Test MLP creation."""
        mlp = MLP(2, [3, 2, 1])
        assert len(mlp.layers) == 3
        assert len(mlp.layers[0].neurons) == 3  # First layer: 3 outputs
        assert len(mlp.layers[1].neurons) == 2  # Second layer: 2 outputs
        assert len(mlp.layers[2].neurons) == 1  # Third layer: 1 output

    def test_mlp_parameters(self):
        """Test MLP parameters."""
        mlp = MLP(2, [3, 1])
        params = mlp.parameters()
        
        # Layer 1: 2 inputs -> 3 outputs = 2*3 + 3 = 9 parameters
        # Layer 2: 3 inputs -> 1 output = 3*1 + 1 = 4 parameters
        # Total: 9 + 4 = 13 parameters
        assert len(params) == 13
        assert all(isinstance(p, Value) for p in params)

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        mlp = MLP(2, [3, 1])
        x = [1.0, 2.0]
        output = mlp(x)
        
        assert isinstance(output, Value)

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP."""
        mlp = MLP(2, [3, 1])
        x = [1.0, 2.0]
        output = mlp(x)
        output.backward()
        
        # All parameters should have gradients
        for param in mlp.parameters():
            assert param.dv != 0

    def test_mlp_repr(self):
        """Test MLP string representation."""
        mlp = MLP(2, [3, 1])
        repr_str = repr(mlp)
        assert 'MLP of' in repr_str

    def test_mlp_zero_grad(self):
        """Test zero_grad on MLP."""
        mlp = MLP(2, [3, 1])
        x = [1.0, 2.0]
        output = mlp(x)
        output.backward()
        
        # Check gradients are non-zero
        assert any(p.dv != 0 for p in mlp.parameters())
        
        # Zero gradients
        mlp.zero_grad()
        assert all(p.dv == 0 for p in mlp.parameters())

    def test_mlp_training_step(self):
        """Test a complete training step."""
        mlp = MLP(2, [3, 1])
        x = [1.0, 2.0]
        target = 0.5
        
        # Forward pass
        output = mlp(x)
        loss = (output - target) ** 2
        
        # Backward pass
        mlp.zero_grad()
        loss.backward()
        
        # Check gradients
        assert any(p.dv != 0 for p in mlp.parameters())
        
        # Update parameters (gradient descent step)
        lr = 0.01
        for param in mlp.parameters():
            param.v -= lr * param.dv
        
        # Check that parameters were updated
        assert any(p.v != 0 for p in mlp.parameters())


class TestMLPGradientChecking:
    """Test MLP gradients using finite differences."""

    def test_neuron_gradient_check(self):
        """Test neuron gradients using finite differences."""
        x = [1.0, 2.0]
        weights = [0.5, -0.3]
        bias = 0.1
        
        # Get the actual gradient from the neuron
        neuron = Neuron(2)
        neuron.weights[0].v = weights[0]
        neuron.weights[1].v = weights[1]
        neuron.bias.v = bias
        output = neuron(x)
        output.backward()
        
        # Check first weight gradient using finite differences
        def f(weight_val):
            temp_neuron = Neuron(2)
            temp_neuron.weights[0].v = weight_val.v
            temp_neuron.weights[1].v = weights[1]
            temp_neuron.bias.v = bias
            return temp_neuron(x)
        
        grad_w0_finite = finite_grad(f, neuron.weights[0])
        
        assert_gradients_close(neuron.weights[0].dv, grad_w0_finite, atol=1e-4)

    def test_mlp_gradient_check(self):
        """Test MLP gradients using finite differences."""
        def mlp_loss(mlp, x, target):
            output = mlp(x)
            return (output - target) ** 2
        
        mlp = MLP(2, [2, 1])
        x = [1.0, 2.0]
        target = 0.5
        
        # Forward pass
        loss = mlp_loss(mlp, x, target)
        loss.backward()
        
        # Check a few parameter gradients
        param = mlp.parameters()[0]  # First parameter
        
        # Finite difference gradient
        def f(param_val):
            temp_mlp = MLP(2, [2, 1])
            # Copy all parameters except the one we're testing
            for i, p in enumerate(mlp.parameters()):
                if i == 0:
                    temp_mlp.parameters()[i].v = param_val.v
                else:
                    temp_mlp.parameters()[i].v = p.v
            return mlp_loss(temp_mlp, x, target)
        
        finite_grad_val = finite_grad(f, param)
        
        # Compare with computed gradient
        assert_gradients_close(param.dv, finite_grad_val, atol=1e-4)

