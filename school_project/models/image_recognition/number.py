import pickle
import gzip

import numpy as np

from school_project.models.utils.perceptron_model import (
                                                       AbstractPerceptronModel
                                                       )

class PerceptronModel(AbstractPerceptronModel):
    """ANN model with no hidden layers,
       that trains to predict Numbers from images."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(output_neuron_count=10, learning_rate=1.0)
    
    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load image input and output datasets.
        
        Returns:
            tuple of image train_inputs, train_outputs,
            test_inputs and test_outputs numpy.ndarrys.
        
        Raises:
            FileNotFoundError: if file does not exist.
        
        """
        # Load datasets from pkl.gz file
        with gzip.open(
              'school_project/models/image_recognition/datasets/mnist.pkl.gz',
              'rb'
              ) as mnist:
            (train_inputs, train_outputs),\
            (test_inputs, test_outputs) = pickle.load(mnist, encoding='bytes')

        # Reshape input arrays into 1 dimension (flatten),
        # then divide by 255 (RGB)
        # to standardize them to a number between 0 and 1
        train_inputs = train_inputs.reshape((train_inputs.shape[0],
                                             -1)).T / 255
        test_inputs = test_inputs.reshape(test_inputs.shape[0], -1).T / 255

        # Represent number values
        # with a one at the matching index of an array of zeros
        train_outputs = np.eye(np.max(train_outputs) + 1)[train_outputs].T
        test_outputs = np.eye(np.max(test_outputs) + 1)[test_outputs].T

        return train_inputs, train_outputs, test_inputs, test_outputs