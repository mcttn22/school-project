import numpy as np

from tools import sigmoid

class ShallowModel():
    """Shallow ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise model values."""

        # Setup model data
        self.train_inputs: np.ndarray = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.train_outputs: np.ndarray = np.array([[0, 1, 1, 0]])
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.hidden_neuron_count: int = 2
        self.output_neuron_count: int = 1
        
        # Setup weights and biases
        np.random.seed(2)  # Sets up pseudo random values for weight arrays
        self.hidden_weights: np.ndarray
        self.output_weights: np.ndarray
        self.hidden_biases: np.ndarray
        self.output_biases: np.ndarray
        self.LEARNING_RATE: float = 0.1

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Number of hidden neurons: {self.hidden_neuron_count}\n" +
                f"Hidden Weights: {self.hidden_weights.tolist()}\n" +
                f"Output Weights: {self.output_weights.tolist()}\n" +
                f"Hidden biases: {self.hidden_biases.tolist()}\n" +
                f"Output biases: {self.output_biases.tolist()}\n" +
                f"Learning Rate: {self.LEARNING_RATE}")

    def init_model_values(self) -> None:
        """Initialise weights to randdom values and biases to 0s"""
        self.hidden_weights = np.random.rand(self.hidden_neuron_count,
                                             self.input_neuron_count)
        self.output_weights = np.random.rand(self.output_neuron_count,
                                             self.hidden_neuron_count)
        self.hidden_biases: np.ndarray = np.zeros(
                                          shape=(self.hidden_neuron_count, 1)
                                          )
        self.output_biases: np.ndarray = np.zeros(
                                          shape=(self.output_neuron_count, 1))

    def back_propagation(self, hidden_output: np.ndarray,
                         prediction: np.ndarray) -> None:
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
        output_weight_gradient: np.ndarray = np.dot(prediction - self.train_outputs, hidden_output.T) / self.train_inputs.shape[1]
        hidden_weight_gradient: np.ndarray = np.dot(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output), self.train_inputs.T) / self.train_inputs.shape[1]
        output_bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.train_inputs.shape[1]
        hidden_bias_gradient: np.ndarray = np.sum(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output)) / self.train_inputs.shape[1]

        # Reshape arrays to match the weight arrays for multiplication
        output_weight_gradient = np.reshape(output_weight_gradient,
                                            self.output_weights.shape)
        hidden_weight_gradient = np.reshape(hidden_weight_gradient,
                                            self.hidden_weights.shape)
        
        # Update weights and biases
        self.output_weights -= self.LEARNING_RATE * output_weight_gradient
        self.hidden_weights -= self.LEARNING_RATE * hidden_weight_gradient
        self.output_biases -= self.LEARNING_RATE * output_bias_gradient
        self.hidden_biases -= self.LEARNING_RATE * hidden_bias_gradient

    def forward_propagation(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a prediction with the weights and biases.
        
        Returns:
            a numpy.ndarray of the hidden output values
            and a numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.hidden_weights, self.train_inputs) + self.hidden_biases
        hidden_output: np.ndarray = sigmoid(z1)
        z2: np.ndarray = np.dot(self.output_weights, hidden_output) + self.output_biases
        prediction: np.ndarray = sigmoid(z2)
        return hidden_output, prediction

    def predict(self) -> None:
        """Use trained weights and biases to predict ouput of XOR gate on two inputs."""
        hidden_output, self.test_prediction = self.forward_propagation()
        
        # Calculate performance of model
        self.test_prediction_accuracy = 100 - np.mean(
                                              np.abs(self.test_prediction
                                                     - self.train_outputs)
                                              ) * 100

    def train(self, epochs: int) -> None:
        """Train weights and biases.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        
        """
        self.train_losses = []
        for epoch in range(epochs):
            if not self.running:
                break
            hidden_output, prediction = self.forward_propagation()
            loss: float = - (1/self.train_inputs.shape[1]) * np.sum(self.train_outputs * np.log(prediction) + (1 - self.train_outputs) * np.log(1 - prediction))
            self.train_losses.append(loss)
            self.back_propagation(hidden_output=hidden_output,
                                  prediction=prediction)
            
class DeepModel():
    """Deep ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise model values."""

        # Setup model data
        self.train_inputs: np.ndarray = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.train_outputs: np.ndarray = np.array([[0, 1, 1, 0]])
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.hidden_neuron_count: int = 2
        self.output_neuron_count: int = 1
        
        # Setup weights and biases
        np.random.seed(2)  # Sets up pseudo random values for weight arrays
        self.hidden_weights: np.ndarray
        self.output_weights: np.ndarray
        self.hidden_biases: np.ndarray
        self.output_biases: np.ndarray
        self.LEARNING_RATE: float = 0.1

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Number of hidden neurons: {self.hidden_neuron_count}\n" +
                f"Hidden Weights: {self.hidden_weights.tolist()}\n" +
                f"Output Weights: {self.output_weights.tolist()}\n" +
                f"Hidden biases: {self.hidden_biases.tolist()}\n" +
                f"Output biases: {self.output_biases.tolist()}\n" +
                f"Learning Rate: {self.LEARNING_RATE}")

    def init_model_values(self) -> None:
        """Initialise weights to randdom values and biases to 0s"""
        self.hidden_weights = np.random.rand(self.hidden_neuron_count,
                                             self.input_neuron_count)
        self.output_weights = np.random.rand(self.output_neuron_count,
                                             self.hidden_neuron_count)
        self.hidden_biases: np.ndarray = np.zeros(
                                          shape=(self.hidden_neuron_count, 1)
                                          )
        self.output_biases: np.ndarray = np.zeros(
                                          shape=(self.output_neuron_count, 1))

    def sigmoid(self, z: np.ndarray | int | float) -> np.ndarray | float:
        """Transfer function, transform input to number between 0 and 1.

        Args:
            z (numpy.ndarray | int | float):
            the numpy.ndarray | int | float to be transferred.
        Returns:
            numpy.ndarray | float,
            with all values | the value transferred to a number between 0-1.
        Raises:
            TypeError: if z is not of type numpy.ndarray | int | float.

        """
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, hidden_output: np.ndarray,
                         prediction: np.ndarray) -> None:
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
        output_weight_gradient: np.ndarray = np.dot(prediction - self.train_outputs, hidden_output.T) / self.train_inputs.shape[1]
        hidden_weight_gradient: np.ndarray = np.dot(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output), self.train_inputs.T) / self.train_inputs.shape[1]
        output_bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.train_inputs.shape[1]
        hidden_bias_gradient: np.ndarray = np.sum(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output)) / self.train_inputs.shape[1]

        # Reshape arrays to match the weight arrays for multiplication
        output_weight_gradient = np.reshape(output_weight_gradient,
                                            self.output_weights.shape)
        hidden_weight_gradient = np.reshape(hidden_weight_gradient,
                                            self.hidden_weights.shape)
        
        # Update weights and biases
        self.output_weights -= self.LEARNING_RATE * output_weight_gradient
        self.hidden_weights -= self.LEARNING_RATE * hidden_weight_gradient
        self.output_biases -= self.LEARNING_RATE * output_bias_gradient
        self.hidden_biases -= self.LEARNING_RATE * hidden_bias_gradient

    def forward_propagation(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a prediction with the weights and biases.
        
        Returns:
            a numpy.ndarray of the hidden output values
            and a numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.hidden_weights, self.train_inputs) + self.hidden_biases
        hidden_output: np.ndarray = sigmoid(z1)
        z2: np.ndarray = np.dot(self.output_weights, hidden_output) + self.output_biases
        prediction: np.ndarray = sigmoid(z2)
        return hidden_output, prediction

    def predict(self) -> None:
        """Use trained weights and biases to predict ouput of XOR gate on two inputs."""
        hidden_output, self.test_prediction = self.forward_propagation()
        
        # Calculate performance of model
        self.test_prediction_accuracy = 100 - np.mean(
                                              np.abs(self.test_prediction
                                                     - self.train_outputs)
                                              ) * 100

    def train(self, epochs: int) -> None:
        """Train weights and biases.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        
        """
        self.train_losses = []
        for epoch in range(epochs):
            if not self.running:
                break
            hidden_output, prediction = self.forward_propagation()
            loss: float = - (1/self.train_inputs.shape[1]) * np.sum(self.train_outputs * np.log(prediction) + (1 - self.train_outputs) * np.log(1 - prediction))
            self.train_losses.append(loss)
            self.back_propagation(hidden_output=hidden_output,
                                  prediction=prediction)