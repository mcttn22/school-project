import h5py
import cupy as cp

from school_project.models.gpu.utils.model import AbstractModel

class Model(AbstractModel):
    """ANN model that trains to predict if an image is a cat or not a cat."""
    def __init__(self, hidden_layers_shape: list[int],
                 learning_rate: float) -> None:
        """Initialise Model's Base class.

        Args:
            hidden_layers_shape (list[int]):
            list of the number of neurons in each hidden layer.
            learning_rate (float): the learning rate of the model.
        
        """
        super().__init__(hidden_layers_shape=hidden_layers_shape,
                         learning_rate=learning_rate)
    
    def load_datasets(self) -> tuple[cp.ndarray, cp.ndarray, 
                                     cp.ndarray, cp.ndarray]:
        """Load image input and output datasets.
        
        Returns:
            tuple of image train_inputs, train_outputs,
            test_inputs and test_outputs cupy.ndarrys.
        
        Raises:
            FileNotFoundError: if file does not exist.
        
        """
        # Load datasets from h5 files
        # (h5 files stores large amount of data with quick access)
        train_dataset: h5py.File = h5py.File(
             r'school_project/models/datasets/train-cat.h5',
             'r'
             )
        test_dataset: h5py.File = h5py.File(
              r'school_project/models/datasets/test-cat.h5',
              'r'
              )
        
        # Load input arrays,
        # containing the RGB values for each pixel in each 64x64 pixel image,
        # for 209 images
        train_inputs: cp.ndarray = cp.array(train_dataset['train_set_x'][:])
        test_inputs: cp.ndarray = cp.array(test_dataset['test_set_x'][:])
        
        # Load output arrays of 1s for cat and 0s for not cat
        train_outputs: cp.ndarray = cp.array(train_dataset['train_set_y'][:])
        test_outputs: cp.ndarray = cp.array(test_dataset['test_set_y'][:])
        
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