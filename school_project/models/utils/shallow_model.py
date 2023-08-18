import numpy as np

from school_project.models.utils.tools import (
                                              ModelInterface,
                                              sigmoid,
                                              calculate_loss,
                                              calculate_prediction_accuracy,
                                              calculate_prediction_correctness
                                              )

class AbstractShallowModel(ModelInterface):
    """ANN model with a single hidden layer"""
    def __init__(self, hidden_neuron_count: int,
                 output_neuron_count: int,
                 learning_rate: float) -> None:
        """Initialise model values.

        Args:
            hidden_neuron_count (int):
            the number of hidden neurons in the model.
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
        self.test_prediction_correctness: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.input_count: int = self.train_inputs.shape[1]
        self.hidden_neuron_count: int = hidden_neuron_count
        self.output_neuron_count: int = output_neuron_count
        
        # Setup weights and biases
        np.random.seed(2)  # Sets up pseudo random values for weight arrays
        self.hidden_weights: np.ndarray
        self.output_weights: np.ndarray
        self.hidden_biases: np.ndarray
        self.output_biases: np.ndarray
        self.learning_rate: float = learning_rate

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
                f"Learning Rate: {self.learning_rate}")

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
        output_weight_gradient: np.ndarray = np.dot(prediction - self.train_outputs, hidden_output.T) / self.input_count
        hidden_weight_gradient: np.ndarray = np.dot(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output), self.train_inputs.T) / self.input_count
        output_bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.input_count
        hidden_bias_gradient: np.ndarray = np.sum(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output)) / self.input_count

        # Reshape arrays to match the weight arrays for multiplication
        output_weight_gradient = np.reshape(output_weight_gradient,
                                            self.output_weights.shape)
        hidden_weight_gradient = np.reshape(hidden_weight_gradient,
                                            self.hidden_weights.shape)
        
        # Update weights and biases
        self.output_weights -= self.learning_rate * output_weight_gradient
        self.hidden_weights -= self.learning_rate * hidden_weight_gradient
        self.output_biases -= self.learning_rate * output_bias_gradient
        self.hidden_biases -= self.learning_rate * hidden_bias_gradient

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
        """Use trained weights and biases to predict."""
        hidden_output, self.test_prediction = self.forward_propagation()
        
        # Calculate performance of model
        self.test_prediction_accuracy = calculate_prediction_accuracy(
                                              prediction=self.test_prediction,
                                              outputs=self.test_outputs
                                              )
        self.test_prediction_correctness = calculate_prediction_correctness(
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
            hidden_output, prediction = self.forward_propagation()
            loss = calculate_loss(input_count=self.input_count,
                                  outputs=self.train_outputs,
                                  prediction=prediction)
            self.train_losses.append(loss)
            self.back_propagation(hidden_output=hidden_output,
                                  prediction=prediction)