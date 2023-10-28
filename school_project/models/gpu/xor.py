import cupy as cp

from school_project.models.gpu.utils.model import AbstractModel

class Model(AbstractModel):
    """ANN model that trains to predict the output of a XOR gate with two 
       inputs."""
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
        """Load XOR input and output datasets.
        
        Returns:
            tuple of XOR train_inputs, train_outputs,
            test_inputs and test_outputs cupy.ndarrys.

        """
        inputs: cp.ndarray = cp.array([[0, 0, 1, 1],
                                       [0, 1, 0, 1]])
        outputs: cp.ndarray = cp.array([[0, 1, 1, 0]])

        return inputs, outputs, inputs, outputs