import unittest
from school_project.models.cpu.utils import tools

class TestTools(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTools, self).__init__(*args, **kwargs)

    def test_sigmoid(self) -> None:
        """Test sigmoid output range to be within 0-1.
        
        Raises:
            AssertionError: if sigmoid output range is not within 0-1.
        
        """
        test_inputs = [-100,0,100]
        for test_input in test_inputs:
            output = tools.sigmoid(test_input)
            self.assertTrue(output >= 0 and output <= 1,
                            "Sigmoid should return a number between 0 and 1")
            
    def test_relu(self) -> None:
        """Test ReLu output range to be >=0.
        
        Raises:
            AssertionError: if relu output range is not >=0.
        
        """
        test_inputs = [-100,0,100]
        for test_input in test_inputs:
            output = tools.relu(test_input)
            self.assertTrue(
                      output >= 0,
                      "ReLu should return a number greater than or equal to 0"
                      )

if __name__ == '__main__':
    unittest.main()