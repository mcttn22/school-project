import numpy as np

from school_project.models.utils.deep_model import AbstractDeepModel

class DeepModel(AbstractDeepModel):
    """Deep ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise Model's Base class."""
        super().__init__(hidden_layers_shape=[2, 2],
                         learning_rate=1)

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