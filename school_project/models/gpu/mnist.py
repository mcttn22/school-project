import pickle
import gzip

import cupy as cp

from school_project.models.gpu.utils.model import (
                                                    AbstractModel
                                                    )

class Model(AbstractModel):
    """ANN model that trains to predict Numbers from images."""
    def __init__(self, hidden_layers_shape: list[int],
                 learning_rate: float,
                 epoch_count: int) -> None:
        """Initialise Model's Base class.

        Args:
            hidden_layers_shape (list[int]):
            list of the number of neurons in each hidden layer.
            learning_rate (float): the learning rate of the model.
            epoch_count (int): the number of training epochs.
        
        """
        super().__init__(hidden_layers_shape=hidden_layers_shape,
                         learning_rate=learning_rate,
                         epoch_count=epoch_count)
    
    def load_datasets(self) -> tuple[cp.ndarray, cp.ndarray, 
                                     cp.ndarray, cp.ndarray]:
        """Load image input and output datasets.
        
        Returns:
            tuple of image train_inputs, train_outputs,
            test_inputs and test_outputs cupy.ndarrys.
        
        Raises:
            FileNotFoundError: if file does not exist.
        
        """
        # Load datasets from pkl.gz file
        with gzip.open(
              'school_project/models/datasets/mnist.pkl.gz',
              'rb'
              ) as mnist:
            (train_inputs, train_outputs),\
            (test_inputs, test_outputs) = pickle.load(mnist, encoding='bytes')

        # Reshape input arrays into 1 dimension (flatten),
        # then divide by 255 (RGB)
        # to standardize them to a number between 0 and 1
        train_inputs = cp.array(train_inputs.reshape((train_inputs.shape[0],
                                             -1)).T / 255)
        test_inputs = cp.array(test_inputs.reshape(test_inputs.shape[0], -1).T / 255)

        # Represent number values
        # with a one at the matching index of an array of zeros
        train_outputs = cp.eye(cp.max(train_outputs) + 1)[train_outputs].T
        test_outputs = cp.eye(cp.max(test_outputs) + 1)[test_outputs].T

        return train_inputs, train_outputs, test_inputs, test_outputs