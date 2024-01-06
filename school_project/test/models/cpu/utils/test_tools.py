"""Unit tests for tools module."""

import unittest

from school_project.models.cpu.utils import tools

class TestTools(unittest.TestCase):
    """Unit tests for the tools module."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialise unit tests."""
        super(TestTools, self).__init__(*args, **kwargs)

    def test_relu(self) -> None:
        """Test ReLu output range to be >=0."""
        test_inputs = [-100, 0, 100]
        for test_input in test_inputs:
            output = tools.relu(z=test_input)
            self.assertGreaterEqual(a=output, b=0)

    def test_sigmoid(self) -> None:
        """Test sigmoid output range to be within 0-1."""
        test_inputs = [-100, 0, 100]
        for test_input in test_inputs:
            output = tools.sigmoid(z=test_input)
            self.assertTrue(expr=output >= 0 and output <= 1)

if __name__ == '__main__':
    unittest.main()