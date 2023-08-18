import unittest
from school_project.models.utils import tools

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

if __name__ == '__main__':
    unittest.main()