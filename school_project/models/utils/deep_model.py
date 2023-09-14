import numpy as np

from school_project.models.utils.tools import (
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
            transfer_type (str): the transfer function
            ('relu' or 'sigmoid')

        """
        self.input: np.ndarray
        self.z: np.ndarray
        # Setup weights and biases
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.init_layer_values(
                               input_neuron_count=input_neuron_count,
                               output_neuron_count=output_neuron_count
                               )
        self.learning_rate = learning_rate
        self.transfer_type = transfer_type

    def __repr__(self) -> str:
        """Read values of the layer.
        
        Returns:
            a string description of the layers's
            weights, bias and learning rate values.

        """
        return (f"Weights: {self.weights.tolist()}\n" +
                f"Biases: {self.biases.tolist()}\n" +
                f"Learning Rate: {self.learning_rate}")

    def init_layer_values(self, input_neuron_count: int, output_neuron_count: int):
        """Initialise weights to randdom values and biases to 0s"""
        self.weights = np.random.rand(output_neuron_count, input_neuron_count)
        self.biases: np.ndarray = np.zeros(
                                            shape=(output_neuron_count, 1)
                                            )

    def back_propagation(self, loss_derivative: np.ndarray) -> None:
        """Adjust the weights and biases via gradient descent.
        
        Args:
            hidden_output (numpy.ndarray): the matrice of hidden output values
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            ValueError:
            if prediction or hidden_output
            is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        if self.transfer_type == 'sigmoid':
            weight_gradient: np.ndarray = np.dot(sigmoid_derivative(self.z) * loss_derivative, self.input.T)
            bias_gradient: np.ndarray = np.sum(sigmoid_derivative(self.z) * loss_derivative)

            loss_derivative = np.dot(self.weights.T, loss_derivative * sigmoid_derivative(self.z))
        
        elif self.transfer_type == 'relu':
            weight_gradient: np.ndarray = np.dot(relu_derivative(self.z) * loss_derivative, self.input.T)
            bias_gradient: np.ndarray = np.sum(relu_derivative(self.z) * loss_derivative)

            loss_derivative = np.dot(self.weights.T, loss_derivative * relu_derivative(self.z))

        # Update weights and biases
        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

        return loss_derivative

    def forward_propagation(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate a prediction with the weights and biases.
        
        Args:
            inputs (np.ndarray): the input values to the layer.
        Returns:
            a numpy.ndarray of the output values

        """
        self.input = inputs
        self.z = np.dot(self.weights, self.input) + self.biases
        if self.transfer_type == 'sigmoid':
            output: np.ndarray = sigmoid(self.z)
        elif self.transfer_type == 'relu':
            output: np.ndarray = relu(self.z)
        return output

class AbstractDeepModel(ModelInterface):
    """ANN model with variable number of hidden layers"""
    def __init__(self, hidden_layers_shape: list[int],
                       learning_rate: float) -> None:
        """Initialise model values.

        Args:
            hidden_layers_shape (list[int]):
            list of the number of neurons in each hidden layer.
            output_neuron_count (int):
            the number of output neurons in the model.
            learning_rate (float): the learning rate of the model.
        
        """
        # Setup model data
        self.train_inputs, self.train_outputs,\
        self.test_inputs, self.test_outputs = self.load_datasets()
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.input_count: int = self.train_inputs.shape[1]
        self.hidden_layers_shape = hidden_layers_shape
        self.output_neuron_count: int = self.train_outputs.shape[0]
        
        # Setup layers
        self.layers: list[FullyConnectedLayer] = []
        np.random.seed(2)  # Sets up pseudo random values for layer weight arrays
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's shape,
            weights, bias and learning rate values.

        """
        return (f"Layers Shape: {','.join(f'{i}' for i in ([self.input_neuron_count] + self.hidden_layers_shape + [self.output_neuron_count]))}\n" +
                f"Learning Rate: {self.learning_rate}")

    def init_model_values(self) -> None:
        """Initialise model layers"""
        # Add input layer
        self.layers.append(FullyConnectedLayer(learning_rate=self.learning_rate,
                                               input_neuron_count=self.input_neuron_count,
                                               output_neuron_count=self.hidden_layers_shape[0],
                                               transfer_type='relu'))

        # Add hidden layers
        for layer in range(len(self.hidden_layers_shape) - 1):
            self.layers.append(FullyConnectedLayer(
                                   learning_rate=self.learning_rate,
                                   input_neuron_count=self.hidden_layers_shape[layer],
                                   output_neuron_count=self.hidden_layers_shape[layer + 1],
                                   transfer_type='relu'))
        
        # Add output layer
        self.layers.append(FullyConnectedLayer(learning_rate=self.learning_rate,
                                               input_neuron_count=self.hidden_layers_shape[-1],
                                               output_neuron_count=self.output_neuron_count,
                                               transfer_type='sigmoid'))

    def back_propagation(self, loss_derivative: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_derivative = layer.back_propagation(loss_derivative)

    def forward_propagation(self) -> np.ndarray:
        output = self.train_inputs
        for layer in self.layers:
            output = layer.forward_propagation(inputs=output)
        return output

    def predict(self) -> None:
        """Use trained weights and biases to predict."""
        output = self.test_inputs
        for layer in self.layers:
            output = layer.forward_propagation(inputs=output)
        self.test_prediction: np.ndarray = output
        
        # Calculate performance of model
        self.test_prediction_accuracy = calculate_prediction_accuracy(
                                              prediction=self.test_prediction,
                                              outputs=self.test_outputs
                                              )

    def train(self, epochs: int) -> None:
        """Train weights and biases.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        
        """
        self.train_losses = []
        for epoch in range(epochs):
            if not self.running:
                break
            prediction = self.forward_propagation()
            loss = calculate_loss(input_count=self.input_count,
                                  outputs=self.train_outputs,
                                  prediction=prediction)
            self.train_losses.append(loss)
            loss_derivative: np.ndarray = -(1/self.input_count) * ((self.train_outputs - prediction)/(prediction * (1 - prediction)))
            self.back_propagation(loss_derivative=loss_derivative)