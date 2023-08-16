import h5py
import numpy as np

from tools import sigmoid

class PerceptronModel():
    """ANN model with no hidden layers,
       that trains to predict if an image is a cat or not a cat."""
    def __init__(self) -> None:
        """Initialise model values."""
        
        # Setup model data
        self.train_inputs, self.train_outputs,\
        self.test_inputs, self.test_outputs = self.load_datasets()
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float

        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.output_neuron_count: int = 1
        
        # Setup weights and bias
        self.weights: np.ndarray
        self.bias: float
        self.LEARNING_RATE: float = 0.001

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Weights: {self.weights}\n" +
                f"Bias: {self.bias}\n" +
                f"Learning Rate: {self.LEARNING_RATE}")
    
    def init_model_values(self) -> None:
        """Initialise weights and bias to 0/s."""
        self.weights = np.zeros(shape=(self.train_inputs.shape[0], 1))
        self.bias = 0
    
    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load image datasets.
        
        Returns:
            image input and output arrays for training and testing.
        Raises:
            FileNotFoundError: if file does not exist.

        """
        
        # Load datasets from h5 files
        # (h5 files stores large amount of data with quick access)
        train_dataset: h5py.File = h5py.File(
                                      r'school_project/datasets/train-cat.h5',
                                      'r'
                                      )
        test_dataset: h5py.File = h5py.File(
                                      r'school_project/datasets/test-cat.h5',
                                      'r'
                                      )
        
        # Load input arrays,
        # containing the RGB values for each pixel in each 64x64 pixel image,
        # for 209 images
        train_inputs: np.ndarray = np.array(train_dataset['train_set_x'][:])
        test_inputs: np.ndarray = np.array(test_dataset['test_set_x'][:])
        
        # Load output arrays of 1s for cat and 0s for not cat
        train_outputs: np.ndarray = np.array(train_dataset['train_set_y'][:])
        test_outputs: np.ndarray = np.array(test_dataset['test_set_y'][:])
        
        # Reshape input arrays into 1 dimension (flatten),
        # then divide by 255 (RGB)
        # to standardize them to a number between 0 and 1
        train_inputs = train_inputs.reshape((train_inputs.shape[0], -1)).T / 255
        test_inputs = test_inputs.reshape((test_inputs.shape[0], -1)).T / 255
        
        # Reshape output arrays into a 1 dimensional list of outputs
        train_outputs = train_outputs.reshape((1, train_outputs.shape[0]))
        test_outputs = test_outputs.reshape((1, test_outputs.shape[0]))
        return train_inputs, train_outputs, test_inputs, test_outputs

    def back_propagation(self, prediction: np.ndarray) -> None:
        """Adjust the weights and bias via gradient descent.
        
        Args:
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            ValueError:
            if prediction is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        weight_gradient: np.ndarray = np.dot(self.train_inputs, (prediction - self.train_outputs).T) / self.train_inputs.shape[1]
        bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.train_inputs.shape[1]
        
        # Update weights and bias
        self.weights -= self.LEARNING_RATE * weight_gradient
        self.bias -= self.LEARNING_RATE * bias_gradient

    def forward_propagation(self) -> np.ndarray:
        """Generate a prediction with the weights and bias.
        
        Returns:
            numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.weights.T, self.train_inputs) + self.bias
        prediction: np.ndarray = sigmoid(z1)
        return prediction

    def predict(self) -> None:
        """Use trained weights and bias
           to predict if image is a cat or not a cat."""
        
        # Calculate prediction for test dataset
        z1: np.ndarray = np.dot(self.weights.T, self.test_inputs) + self.bias
        self.test_prediction = sigmoid(z1)
        
        # Calculate performance of model
        self.test_prediction_accuracy = 100 - np.mean(
                                              np.abs(
                                                  self.test_prediction.round()
                                                  - self.test_outputs
                                                  )
                                              ) * 100

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
            loss: float = - (1/self.train_inputs.shape[1]) * np.sum(self.train_outputs * np.log(prediction) + (1 - self.train_outputs) * np.log(1 - prediction))
            self.train_losses.append(np.squeeze(loss))
            self.back_propagation(prediction=prediction)