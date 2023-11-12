import time

import numpy as np

from school_project.models.cpu.utils.tools import (
                                              ModelInterface,
                                              relu,
                                              relu_derivative,
                                              sigmoid,
                                              sigmoid_derivative,
                                              calculate_loss,
                                              calculate_prediction_accuracy
                                              )

class _FullyConnectedLayer():
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
        self.input: np.ndarray
        self.output: np.ndarray

        # Setup weights and biases
        self.weights: np.ndarray
        self.biases: np.ndarray
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
        np.random.seed(1)  # Sets up pseudo random values for layer weight arrays
        self.weights = np.random.rand(self.output_neuron_count, self.input_neuron_count) - 0.5
        self.biases = np.zeros(shape=(self.output_neuron_count, 1))

    def init_layer_values_zeros(self) -> None:
        """Initialise weights to 0s and biases to 0s"""
        self.weights = np.zeros(shape=(self.output_neuron_count, self.input_neuron_count))
        self.biases = np.zeros(shape=(self.output_neuron_count, 1))

    def back_propagation(self, dloss_doutput) -> np.ndarray:
        """Adjust the weights and biases via gradient descent.
        
        Args:
            dloss_doutput (numpy.ndarray): the derivative of the loss of the 
            layer's output, with respect to the layer's output.
        Returns:
            a numpy.ndarray derivative of the loss of the layer's input,
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

        dloss_dweights = np.dot(dloss_dz, self.input.T)
        dloss_dbiases = np.sum(dloss_dz)
        
        dloss_dinput = np.dot(self.weights.T, dloss_dz)

        # Update weights and biases
        self.weights -= self.learning_rate * dloss_dweights
        self.biases -= self.learning_rate * dloss_dbiases

        return dloss_dinput

    def forward_propagation(self, inputs) -> np.ndarray:
        """Generate a layer output with the weights and biases.
        
        Args:
            inputs (np.ndarray): the input values to the layer.
        Returns:
            a numpy.ndarray of the output values.

        """
        self.input = inputs
        z = np.dot(self.weights, self.input) + self.biases
        if self.transfer_type == 'sigmoid':
            self.output = sigmoid(z)
        elif self.transfer_type == 'relu':
            self.output = relu(z)
        return self.output

class AbstractModel(ModelInterface):
    """ANN model with variable number of hidden layers"""
    def __init__(self, 
                 hidden_layers_shape: list[int],
                 train_dataset_size: int,
                 learning_rate: float,
                 use_relu: bool) -> None:
        """Initialise model values.

        Args:
            hidden_layers_shape (list[int]):
            list of the number of neurons in each hidden layer.
            train_dataset_size (int): the number of train dataset inputs to use.
            learning_rate (float): the learning rate of the model.
            use_relu (bool): True or False whether the ReLu Transfer function 
            should be used.
        
        """
        # Setup model data
        self.train_inputs, self.train_outputs,\
        self.test_inputs, self.test_outputs = self.load_datasets(
                                         train_dataset_size=train_dataset_size
                                         )
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float
        self.training_progress: str = ""
        self.training_time: float
        
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
        self.use_relu = use_relu
        
        # Setup model values
        self.layers: list[_FullyConnectedLayer] = []
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's shape,
            weights, bias and learning rate values.

        """
        return (f"Layers Shape: {','.join(self.layers_shape)}\n" +
                f"Learning Rate: {self.learning_rate}")

    def setup_layers(self) -> None:
        """Setup model layers"""
        # Check if setting up Deep Network
        if len(self.hidden_layers_shape) > 0:
            if self.use_relu:

                # Add input layer
                self.layers.append(_FullyConnectedLayer(
                                        learning_rate=self.learning_rate,
                                        input_neuron_count=self.input_neuron_count,
                                        output_neuron_count=self.hidden_layers_shape[0],
                                        transfer_type='relu'
                                        ))

                # Add hidden layers
                for layer in range(len(self.hidden_layers_shape) - 1):
                    self.layers.append(_FullyConnectedLayer(
                                learning_rate=self.learning_rate,
                                input_neuron_count=self.hidden_layers_shape[layer],
                                output_neuron_count=self.hidden_layers_shape[layer + 1],
                                transfer_type='relu'
                                ))
            else:

                # Add input layer
                self.layers.append(_FullyConnectedLayer(
                                        learning_rate=self.learning_rate,
                                        input_neuron_count=self.input_neuron_count,
                                        output_neuron_count=self.hidden_layers_shape[0],
                                        transfer_type='sigmoid'
                                        ))

                # Add hidden layers
                for layer in range(len(self.hidden_layers_shape) - 1):
                    self.layers.append(_FullyConnectedLayer(
                                learning_rate=self.learning_rate,
                                input_neuron_count=self.hidden_layers_shape[layer],
                                output_neuron_count=self.hidden_layers_shape[layer + 1],
                                transfer_type='sigmoid'
                                ))
            
            # Add output layer
            self.layers.append(_FullyConnectedLayer(
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
            self.layers.append(_FullyConnectedLayer(
                                    learning_rate=self.learning_rate,
                                    input_neuron_count=self.input_neuron_count,
                                    output_neuron_count=self.output_neuron_count,
                                    transfer_type='sigmoid'
                                    ))

            # Initialise Layer values to zeros
            for layer in self.layers:
                layer.init_layer_values_zeros()

    def init_random_values(self) -> None:
        """Initialise model layers"""
        self.setup_layers()

        # Check if setting up Deep Network
        if len(self.hidden_layers_shape) > 0:

            # Initialise Layer values to random values
            for layer in self.layers:
                layer.init_layer_values_random()
        
        # Setup Perceptron Network
        else:

            # Initialise Layer values to zeros
            for layer in self.layers:
                layer.init_layer_values_zeros()

    def load_model_values(self, file_location: str) -> None:
        """Load weights and bias/biases from .npz file.
        
        Args:
            file_location (str): the location of the file to load from.
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        self.setup_layers()

        data: dict[str, np.ndarray] = np.load(file=file_location)

        # Initialise Layer values
        i = 0
        keys = list(data.keys())
        for layer in self.layers:
            layer.weights = data[keys[i]]
            layer.biases = data[keys[i + 1]]
            i += 2

    def back_propagation(self, dloss_doutput) -> None:
        """Train each layer's weights and biases.
        
        Args:
            dloss_doutput (np.ndarray): the derivative of the loss of the 
            output layer's output, with respect to the output layer's output.

        """
        for layer in reversed(self.layers):
            dloss_doutput = layer.back_propagation(dloss_doutput=dloss_doutput)

    def forward_propagation(self) -> np.ndarray:
        """Generate a prediction with the layers.
        
        Returns:
            a numpy.ndarray of the prediction values.

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

    def train(self, epoch_count: int) -> None:
        """Train layers' weights and biases.
        
           Args:
               epoch_count (int): the number of training epochs.

        """
        self.layers_shape = [f'{layer}' for layer in (
                            [self.input_neuron_count] + 
                            self.hidden_layers_shape + 
                            [self.output_neuron_count]
                            )]
        self.train_losses = []
        training_start_time = time.time()
        for epoch in range(epoch_count):
            if not self.running:
                break
            self.training_progress = f"Epoch {epoch} / {epoch_count}"
            prediction = self.forward_propagation()
            loss = calculate_loss(input_count=self.input_count,
                                  outputs=self.train_outputs,
                                  prediction=prediction)
            self.train_losses.append(loss)
            if not self.running:
                break
            dloss_doutput = -(1/self.input_count) * ((self.train_outputs - prediction)/(prediction * (1 - prediction)))
            self.back_propagation(dloss_doutput=dloss_doutput)
        self.training_time = round(number=time.time() - training_start_time,
                                   ndigits=2)
        
    def save_model_values(self, file_location: str) -> None:
        """Save the model by saving the weights then biases of each layer to 
           a .npz file with a given file location.

           Args:
               file_location (str): the file location to save the model to.

        """
        saved_model: list[np.ndarray] = []
        for layer in self.layers:
            saved_model.append(layer.weights)
            saved_model.append(layer.biases)
        np.savez(file_location, *saved_model)