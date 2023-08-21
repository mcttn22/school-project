import numpy as np

class ModelInterface():
    """Interface for ANN models."""
    def init_model_values(self) -> None:
        """Initialise weights and bias/biases.
        
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def load_datasets() -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load input and output datasets.
        
        Returns:
            tuple of train_inputs, train_outputs,
            test_inputs and test_outputs.
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def back_propagation(self, prediction: np.ndarray) -> None:
        """Adjust the weights and bias/biases via gradient descent.
        
        Args:
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            NotImplementedError: if this method is not implemented.
        
        """
        raise NotImplementedError

    def forward_propagation(self) -> np.ndarray:
        """Generate a prediction with the weights and bias/biases.
        
        Returns:
            numpy.ndarray of prediction values.
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def predict(self) -> None:
        """Use trained weights and bias/biases to predict.
           
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def train(self, epochs: int) -> None:
        """Train weights and bias/biases.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        Raises:
            NotImplementedError: if this method is not implemented.
        
        """
        raise NotImplementedError

def sigmoid(z: np.ndarray | int | float) -> np.ndarray | float:
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

def calculate_loss(input_count: int,
                   outputs: np.ndarray,
                   prediction: np.ndarray) -> float:
    """Calculate average loss/error of the prediction to the outputs.
    
    Args:
        input_count (int): the number of inputs.
        outputs (np.ndarray):
        the train/test outputs array to compare with the prediction.
        prediction (np.ndarray): the array of prediction values.
    Returns:
        float loss.
    Raises:
        ValueError:
        if outputs is not a suitable multiplier with the prediction
        (incorrect shapes)

    """
    return np.squeeze(- (1/input_count) * np.sum(outputs * np.log(prediction) + (1 - outputs) * np.log(1 - prediction)))

def calculate_prediction_accuracy(prediction: np.ndarray,
                                  outputs: np.ndarray) -> float:
    """Calculate the percentage accuracy of the predictions.
    
    Args:
        prediction (np.ndarray): the array of prediction values.
        outputs (np.ndarray):
        the train/test outputs array to compare with the prediction.
    Returns:
        float prediction accuracy

    """
    return 100 - np.mean(np.abs(prediction - outputs)) * 100