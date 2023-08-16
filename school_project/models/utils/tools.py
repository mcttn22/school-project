import numpy as np

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