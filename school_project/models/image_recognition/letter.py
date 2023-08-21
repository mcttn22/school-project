import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np

from school_project.models.utils.perceptron_model import (
                                                       AbstractPerceptronModel
                                                       )

class PerceptronModel(AbstractPerceptronModel):
    """ANN model with no hidden layers,
       that trains to predict Numbers from images."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(output_neuron_count=10, learning_rate=0.2)
    
    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load image input and output datasets.
        
        Returns:
            tuple of image train_inputs, train_outputs,
            test_inputs and test_outputs numpy.ndarrys.
        
        Raises:
            FileNotFoundError: if file does not exist.
        
        """
        (train_inputs, train_outputs), (test_inputs, test_outputs) = mnist.load_data()

        train_inputs = train_inputs.reshape((train_inputs.shape[0], -1))
        train_inputs = train_inputs.astype('float32')
        train_inputs = train_inputs.T / 255

        train_outputs = to_categorical(train_outputs).T

        test_inputs = test_inputs.reshape(test_inputs.shape[0], -1)
        test_inputs = test_inputs.astype('float32')
        test_inputs = test_inputs.T / 255
        test_outputs = np.array(to_categorical(test_outputs)).T

        return train_inputs, train_outputs, test_inputs, test_outputs