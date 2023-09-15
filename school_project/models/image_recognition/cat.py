import h5py
import numpy as np

from school_project.models.utils.deep_model import AbstractDeepModel

class DeepModel(AbstractDeepModel):
    """Deep ANN model
       that trains to predict if an image is a cat or not a cat."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(hidden_layers_shape=[100], learning_rate=0.001)
    
    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load image input and output datasets.
        
        Returns:
            tuple of image train_inputs, train_outputs,
            test_inputs and test_outputs numpy.ndarrys.
        
        Raises:
            FileNotFoundError: if file does not exist.
        
        """
        # Load datasets from h5 files
        # (h5 files stores large amount of data with quick access)
        train_dataset: h5py.File = h5py.File(
             r'school_project/models/image_recognition/datasets/train-cat.h5',
             'r'
             )
        test_dataset: h5py.File = h5py.File(
              r'school_project/models/image_recognition/datasets/test-cat.h5',
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
        train_inputs = train_inputs.reshape((train_inputs.shape[0],
                                             -1)).T / 255
        test_inputs = test_inputs.reshape((test_inputs.shape[0], -1)).T / 255
        
        # Reshape output arrays into a 1 dimensional list of outputs
        train_outputs = train_outputs.reshape((1, train_outputs.shape[0]))
        test_outputs = test_outputs.reshape((1, test_outputs.shape[0]))
        return train_inputs, train_outputs, test_inputs, test_outputs