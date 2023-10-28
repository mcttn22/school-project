import cupy as cp

from school_project.models.gpu.utils.tools import (
                                              ModelInterface,
                                              relu,
                                              relu_derivative,
                                              sigmoid,
                                              sigmoid_derivative,
                                              calculate_loss,
                                              calculate_prediction_accuracy
                                              )

class FullyConnectedLayer():
    "Fully connected layer for Deep ANNs"
    def __init__(self, learning_rate: float, input_neuron_count: int,
                 output_neuron_count: int, transfer_type: str) -> None:
        """Initialise layer values.

        Args:
            learning_rate (float): the learning rate of the model.
            input_neuron_count (int):
            the number of input neurons into the layer.
            output_neuron_count (int):
            the number of output neurons into the layer.
            transfer_type (str): the transfer function type
            ('sigmoid' or 'relu')

        """
        # Setup layer attributes
        self.input_neuron_count = input_neuron_count
        self.output_neuron_count = output_neuron_count
        self.transfer_type = transfer_type
        self.input: cp.ndarray
        self.output: cp.ndarray

        # Setup weights and biases
        self.weights: cp.ndarray
        self.biases: cp.ndarray
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        """Read values of the layer.
        
        Returns:
            a string description of the layers's
            weights, bias and learning rate values.

        """
        return (f"Weights: {self.weights.tolist()}\n" +
                f"Biases: {self.biases.tolist()}\n")

    def init_layer_values_random(self) -> None:
        """Initialise weights to random values and biases to 0s"""
        cp.random.seed(2)  # Sets up pseudo random values for layer weight arrays
        self.weights = cp.random.rand(self.output_neuron_count, self.input_neuron_count) - 0.5
        self.biases = cp.zeros(shape=(self.output_neuron_count, 1))

    def init_layer_values_zeros(self) -> None:
        """Initialise weights to 0s and biases to 0s"""
        self.weights = cp.zeros(shape=(self.output_neuron_count, self.input_neuron_count))
        self.biases = cp.zeros(shape=(self.output_neuron_count, 1))

    def back_propagation(self, dloss_doutput) -> cp.ndarray:
        """Adjust the weights and biases via gradient descent.
        
        Args:
            dloss_doutput (cupy.ndarray): the derivative of the loss of the 
            layer's output, with respect to the layer's output.
        Returns:
            a cupy.ndarray derivative of the loss of the layer's input,
            with respect to the layer's input.
        Raises:
            ValueError:
            if dloss_doutput
            is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        if self.transfer_type == 'sigmoid':
            dloss_dz = dloss_doutput * sigmoid_derivative(output=self.output)
        elif self.transfer_type == 'relu':
            dloss_dz = dloss_doutput * relu_derivative(output=self.output)

        dloss_dweights = cp.dot(dloss_dz, self.input.T)
        dloss_dbiases = cp.sum(dloss_dz)
        
        dloss_dinput = cp.dot(self.weights.T, dloss_dz)

        # Update weights and biases
        self.weights -= self.learning_rate * dloss_dweights
        self.biases -= self.learning_rate * dloss_dbiases

        return dloss_dinput

    def forward_propagation(self, inputs) -> cp.ndarray:
        """Generate a layer output with the weights and biases.
        
        Args:
            inputs (cp.ndarray): the input values to the layer.
        Returns:
            a cupy.ndarray of the output values.

        """
        self.input = inputs
        z = cp.dot(self.weights, self.input) + self.biases
        if self.transfer_type == 'sigmoid':
            self.output = sigmoid(z)
        elif self.transfer_type == 'relu':
            self.output = relu(z)
        return self.output

class AbstractModel (ModelInterface):
    """ANN model with variable number of hidden layers"""
    def __init__(self, hidden_layers_shape: list[int],
                       learning_rate: float,
                       epoch_count: int) -> None:
        """Initialise model values.

        Args:
            hidden_layers_shape (list[int]):
            list of the number of neurons in each hidden layer.
            output_neuron_count (int):
            the number of output neurons in the model.
            learning_rate (float): the learning rate of the model.
            epoch_count (int): the number of training epochs.
        
        """
        # Setup model data
        self.train_inputs, self.train_outputs,\
        self.test_inputs, self.test_outputs = self.load_datasets()
        self.train_losses: list[float]
        self.test_prediction: cp.ndarray
        self.test_prediction_accuracy: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.input_count: int = self.train_inputs.shape[1]
        self.hidden_layers_shape = hidden_layers_shape
        self.output_neuron_count: int = self.train_outputs.shape[0]
        self.layers_shape: list[chr] = [f'{layer}' for layer in (
                                                    [self.input_neuron_count] + 
                                                    self.hidden_layers_shape + 
                                                    [self.output_neuron_count]
                                                    )]
        
        # Setup model values
        self.layers: list[FullyConnectedLayer] = []
        self.learning_rate = learning_rate
        self.epoch_count = epoch_count
        self.init_model_values()

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's shape,
            weights, bias and learning rate values.

        """
        return (f"Layers Shape: {','.join(self.layers_shape)}\n" +
                f"Learning Rate: {self.learning_rate}")

    def init_model_values(self) -> None:
        """Initialise model layers"""
        # Check if setting up Deep Network
        if len(self.hidden_layers_shape) > 0:

            # Add input layer
            self.layers.append(FullyConnectedLayer(
                                    learning_rate=self.learning_rate,
                                    input_neuron_count=self.input_neuron_count,
                                    output_neuron_count=self.hidden_layers_shape[0],
                                    transfer_type='relu'
                                    ))

            # Add hidden layers
            for layer in range(len(self.hidden_layers_shape) - 1):
                self.layers.append(FullyConnectedLayer(
                            learning_rate=self.learning_rate,
                            input_neuron_count=self.hidden_layers_shape[layer],
                            output_neuron_count=self.hidden_layers_shape[layer + 1],
                            transfer_type='relu'
                            ))
            
            # Add output layer
            self.layers.append(FullyConnectedLayer(
                                    learning_rate=self.learning_rate,
                                    input_neuron_count=self.hidden_layers_shape[-1],
                                    output_neuron_count=self.output_neuron_count,
                                    transfer_type='sigmoid'
                                    ))

            # Initialise Layer values to random values
            for layer in self.layers:
                layer.init_layer_values_random()

        # Setup Perceptron Network
        else:
            self.layers.append(FullyConnectedLayer(
                                    learning_rate=self.learning_rate,
                                    input_neuron_count=self.input_neuron_count,
                                    output_neuron_count=self.output_neuron_count,
                                    transfer_type='sigmoid'
                                    ))

            # Initialise Layer values to zeros
            for layer in self.layers:
                layer.init_layer_values_zeros()

    def back_propagation(self, dloss_doutput) -> None:
        """Train each layer's weights and biases.
        
        Args:
            dloss_doutput (cp.ndarray): the derivative of the loss of the 
            output layer's output, with respect to the output layer's output.

        """
        for layer in reversed(self.layers):
            dloss_doutput = layer.back_propagation(dloss_doutput=dloss_doutput)

    def forward_propagation(self) -> cp.ndarray:
        """Generate a prediction with the layers.
        
        Returns:
            a cupy.ndarray of the prediction values.

        """
        output = self.train_inputs
        for layer in self.layers:
            output = layer.forward_propagation(inputs=output)
        return output

    def test(self) -> None:
        """Test the layers' trained weights and biases."""
        output = self.test_inputs
        for layer in self.layers:
            output = layer.forward_propagation(inputs=output)
        self.test_prediction = output
        
        # Calculate performance of model
        self.test_prediction_accuracy = calculate_prediction_accuracy(
                                              prediction=self.test_prediction,
                                              outputs=self.test_outputs
                                              )

    def train(self) -> None:
        """Train layers' weights and biases."""
        self.layers_shape = [f'{layer}' for layer in (
                            [self.input_neuron_count] + 
                            self.hidden_layers_shape + 
                            [self.output_neuron_count]
                            )]
        self.train_losses = []
        for epoch in range(self.epoch_count):
            if not self.running:
                break
            prediction = self.forward_propagation()
            loss = calculate_loss(input_count=self.input_count,
                                  outputs=self.train_outputs,
                                  prediction=prediction)
            self.train_losses.append(loss)
            dloss_doutput = -(1/self.input_count) * ((self.train_outputs - prediction)/(prediction * (1 - prediction)))
            self.back_propagation(dloss_doutput=dloss_doutput)