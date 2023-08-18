import numpy as np

from school_project.models.utils.shallow_model import AbstractShallowModel

class ShallowModel(AbstractShallowModel):
    """Shallow ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(hidden_neuron_count=2, output_neuron_count=1, learning_rate=0.1)

    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load XOR input and output datasets.
        
        Returns:
            tuple of XOR train_inputs, train_outputs,
            test_inputs and test_outputs numpy.ndarrys.

        """
        inputs: np.ndarray = np.array([[0, 0, 1, 1],
                                       [0, 1, 0, 1]])
        outputs: np.ndarray = np.array([[0, 1, 1, 0]])

        return inputs, outputs, inputs, outputs
    
class DeepModel(AbstractShallowModel):
    """Deep ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(hidden_neuron_count=2, output_neuron_count=1, learning_rate=0.1)

    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load XOR input and output datasets.
        
        Returns:
            tuple of XOR train_inputs, train_outputs,
            test_inputs and test_outputs numpy.ndarrys.

        """
        inputs: np.ndarray = np.array([[0, 0, 1, 1],
                                       [0, 1, 0, 1]])
        outputs: np.ndarray = np.array([[0, 1, 1, 0]])

        return inputs, outputs, inputs, outputs