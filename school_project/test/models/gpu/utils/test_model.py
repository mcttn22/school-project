"""Unit tests for model module."""

import os
import unittest
import uuid

import cupy as cp

# Test XOR implementation of Model for its lesser computation time
from school_project.models.gpu.xor import XORModel

class TestModel(unittest.TestCase):
    """Unit tests for model module."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialise unit tests and inputs."""
        super(TestModel, self).__init__(*args, **kwargs)

    def test_train_dataset_size(self) -> None:
        """Test the size of training dataset to be value chosen."""
        train_dataset_size = 4
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = train_dataset_size,
                         learning_rate = 0.1,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        self.assertEqual(first=model.layers.head.input.shape[1],
                         second=train_dataset_size)

    def test_network_shape(self) -> None:
        """Test the neuron count of each layer to match the set shape of the
           network."""
        layers_shape = [2, 100, 100, 1]
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = 4,
                         learning_rate = 0.1,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for count, layer in enumerate(model.layers):
            self.assertEqual(first=layer.input_neuron_count,
                             second=layers_shape[count])

    def test_learning_rates(self) -> None:
        """Test learning rate of each layer to be the same."""
        learning_rate = 0.1
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = 4,
                         learning_rate = learning_rate,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for layer in model.layers:
            self.assertEqual(first=layer.learning_rate, second=learning_rate)

    def test_relu_model_transfer_types(self) -> None:
        """Test transfer type of each layer to match whats set."""
        transfer_types = ['relu', 'relu', 'sigmoid']
        model = XORModel(hidden_layers_shape = [100, 100],
                              train_dataset_size = 4,
                              learning_rate = 0.1,
                              use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for count, layer in enumerate(model.layers):
            self.assertEqual(first=layer.transfer_type,
                             second=transfer_types[count])

    def test_sigmoid_model_transfer_types(self) -> None:
        """Test transfer type of each layer to match whats set."""
        transfer_types = ['sigmoid', 'sigmoid', 'sigmoid']
        model = XORModel(hidden_layers_shape = [100, 100],
                                 train_dataset_size = 4,
                                 learning_rate = 0.1,
                                 use_relu = False)
        model.create_model_values()
        model.train(epoch_count=1)
        for count, layer in enumerate(model.layers):
            self.assertEqual(first=layer.transfer_type,
                             second=transfer_types[count])

    def test_weight_matrice_shapes(self) -> None:
        """Test that each layer's weight matrix has the same number of columns
        as the layer's input matrix's number of rows, for the matrice
        multiplication."""
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = 4,
                         learning_rate = 0.1,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for layer in model.layers:
            self.assertEqual(first=layer.weights.shape[1],
                             second=layer.input.shape[0])

    def test_bias_matrice_shapes(self) -> None:
        """Test that each layer's bias matrix has the same number of rows
        as the result of the layer's weights and input multiplication, for
        element-wise addition of the biases."""
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = 4,
                         learning_rate = 0.1,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for layer in model.layers:
            self.assertEqual(first=layer.biases.shape[0],
                             second=layer.weights.shape[0])

    def test_layer_output_shapes(self) -> None:
        """Test the shape of each layer's activation function's output."""
        model = XORModel(hidden_layers_shape = [100, 100],
                         train_dataset_size = 4,
                         learning_rate = 0.1,
                         use_relu = True)
        model.create_model_values()
        model.train(epoch_count=1)
        for layer in model.layers:
            self.assertEqual(
                         first=(layer.weights.shape[0], layer.input.shape[1]),
                         second=layer.output.shape)
            
    def test_save_model(self) -> None:
        """Test that the weights and biases are saved correctly."""
        initial_model = XORModel(hidden_layers_shape = [100, 100],
                                 train_dataset_size = 4,
                                 learning_rate = 0.1,
                                 use_relu = True)
        initial_model.create_model_values()
        initial_model.train(epoch_count=1)

        # Save model values
        file_location = f"school_project/saved-models/{uuid.uuid4().hex}.npz"
        initial_model.save_model_values(file_location=file_location)

        # Create model from the saved values
        loaded_model = XORModel(hidden_layers_shape = [100, 100],
                                train_dataset_size = 4,
                                learning_rate = 0.1,
                                use_relu = True)
        loaded_model.load_model_values(file_location=file_location)

        # Remove the saved model values
        os.remove(path=file_location)

        # Compare initial and loaded model values
        for layer1, layer2 in zip(initial_model.layers, loaded_model.layers):
            self.assertTrue(cp.array_equal(a1=layer1.weights,
                                           a2=layer2.weights))
            self.assertTrue(cp.array_equal(a1=layer1.biases,
                                           a2=layer2.biases))

if __name__ == '__main__':
    unittest.main()
