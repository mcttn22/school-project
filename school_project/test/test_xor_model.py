import unittest
from school_project.xor_model import XorModel

class TestXorModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestXorModel, self).__init__(*args, **kwargs)
        self.xorModel = XorModel()

    def test_sigmoid(self) -> None:
        "Test sigmoid output range to be within 0-1"
        inputs = [-100,0,100]
        for i in inputs:
            output = self.xorModel.sigmoid(i)
            self.assertTrue(output >= 0 and output <= 1, "Sigmoid should return a number between 0 and 1")

if __name__ == '__main__':
    unittest.main()