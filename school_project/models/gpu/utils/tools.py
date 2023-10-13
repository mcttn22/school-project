import cupy as cp
import numpy as np

class ModelInterface():
    """Interface for ANN models."""
    def init_model_values(self) -> None:
        """Initialise weights and bias/biases.
        
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def load_datasets() -> tuple[cp.ndarray, cp.ndarray, 
                                     cp.ndarray, cp.ndarray]:
        """Load input and output datasets.
        
        Returns:
            tuple of train_inputs, train_outputs,
            test_inputs and test_outputs.
        Raises:
            NotImplementedError: if this method is not implemented.

        """
        raise NotImplementedError

    def back_propagation(self, prediction: cp.ndarray) -> None:
        """Adjust the weights and bias/biases via gradient descent.
        
        Args:
            prediction (cupy.ndarray): the matrice of prediction values
        Raises:
            NotImplementedError: if this method is not implemented.
        
        """
        raise NotImplementedError

    def forward_propagation(self) -> cp.ndarray:
        """Generate a prediction with the weights and bias/biases.
        
        Returns:
            cupy.ndarray of prediction values.
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

def relu(z: cp.ndarray | int | float) -> cp.ndarray | float:
    """Transfer function, transform input to max number between 0 and z.

    Args:
        z (cupy.ndarray | int | float):
        the cupy.ndarray | int | float to be transferred.
    Returns:
        cupy.ndarray | float,
        with all values | the value transferred to max number between 0-z.
    Raises:
        TypeError: if z is not of type cupy.ndarray | int | float.

    """
    return cp.maximum(0.1*z, 0)  # Divide by 10 to stop overflow errors

def relu_derivative(output: cp.ndarray | int | float) -> cp.ndarray | float:
    """Calculate derivative of ReLu Transfer function with respect to z.

    Args:
        output (cupy.ndarray | int | float):
        the cupy.ndarray | int | float output of the ReLu transfer function.
    Returns:
        cupy.ndarray | float,
        derivative of the ReLu transfer function with respect to z.
    Raises:
        TypeError: if output is not of type cupy.ndarray | int | float.

    """
    output[output <= 0] = 0
    output[output > 0] = 1
    
    return output

def sigmoid(z: cp.ndarray | int | float) -> cp.ndarray | float:
    """Transfer function, transform input to number between 0 and 1.

    Args:
        z (cupy.ndarray | int | float):
        the cupy.ndarray | int | float to be transferred.
    Returns:
        cupy.ndarray | float,
        with all values | the value transferred to a number between 0-1.
    Raises:
        TypeError: if z is not of type cupy.ndarray | int | float.

    """
    return 1 / (1 + cp.exp(-z))

def sigmoid_derivative(output: cp.ndarray | int | float) -> cp.ndarray | float:
    """Calculate derivative of sigmoid Transfer function with respect to z.

    Args:
        output (cupy.ndarray | int | float):
        the cupy.ndarray | int | float output of the sigmoid transfer function.
    Returns:
        cupy.ndarray | float,
        derivative of the sigmoid transfer function with respect to z.
    Raises:
        TypeError: if output is not of type cupy.ndarray | int | float.

    """
    return output * (1 - output)

def calculate_loss(input_count: int,
                   outputs: cp.ndarray,
                   prediction: cp.ndarray) -> float:
    """Calculate average loss/error of the prediction to the outputs.
    
    Args:
        input_count (int): the number of inputs.
        outputs (cp.ndarray):
        the train/test outputs array to compare with the prediction.
        prediction (cp.ndarray): the array of prediction values.
    Returns:
        float loss.
    Raises:
        ValueError:
        if outputs is not a suitable multiplier with the prediction
        (incorrect shapes)

    """
    return np.squeeze(cp.asnumpy(- (1/input_count) * cp.sum(outputs * cp.log(prediction) + (1 - outputs) * cp.log(1 - prediction))))

def calculate_prediction_accuracy(prediction: cp.ndarray,
                                  outputs: cp.ndarray) -> float:
    """Calculate the percentage accuracy of the predictions.
    
    Args:
        prediction (cp.ndarray): the array of prediction values.
        outputs (cp.ndarray):
        the train/test outputs array to compare with the prediction.
    Returns:
        float prediction accuracy

    """
    return 100 - np.mean(np.abs(cp.asnumpy(prediction - outputs))) * 100