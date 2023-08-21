import numpy as np

from school_project.models.utils.tools import (
                                              ModelInterface, 
                                              sigmoid,
                                              calculate_loss,
                                              calculate_prediction_accuracy
                                              )

class AbstractPerceptronModel(ModelInterface):
    """ANN model with no hidden layers"""
    def __init__(self, output_neuron_count: int, 
                 learning_rate: float) -> None:
        """Initialise model values.

        Args:
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
        self.output_neuron_count = output_neuron_count
        
        # Setup weights and bias
        self.weights: np.ndarray
        self.bias: float
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Weights: {self.weights}\n" +
                f"Bias: {self.bias}\n" +
                f"Learning Rate: {self.learning_rate}")
    
    def init_model_values(self) -> None:
        """Initialise weights and bias to 0/s."""
        self.weights = np.zeros(shape=(self.input_neuron_count,
                                       self.output_neuron_count))
        self.bias = 0

    def back_propagation(self, prediction: np.ndarray) -> None:
        """Adjust the weights and bias via gradient descent.
        
        Args:
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            ValueError:
            if prediction is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        weight_gradient: np.ndarray = np.dot(self.train_inputs, (prediction - self.train_outputs).T) / self.input_count
        bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.input_count
        
        # Update weights and bias
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

    def forward_propagation(self) -> np.ndarray:
        """Generate a prediction with the weights and bias.
        
        Returns:
            numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.weights.T, self.train_inputs) + self.bias
        prediction: np.ndarray = sigmoid(z1)
        return prediction

    def predict(self) -> None:
        """Use trained weights and bias to predict."""
        
        # Calculate prediction for test dataset
        z1: np.ndarray = np.dot(self.weights.T, self.test_inputs) + self.bias
        self.test_prediction = sigmoid(z1)
        
        # Calculate performance of model
        self.test_prediction_accuracy = calculate_prediction_accuracy(
                                              prediction=self.test_prediction,
                                              outputs=self.test_outputs
                                              )

    def train(self, epochs: int) -> None:
        """Train weights and bias.
        
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
            self.back_propagation(prediction=prediction)