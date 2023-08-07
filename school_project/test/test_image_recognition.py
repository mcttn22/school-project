import unittest
from school_project.image_recognition import CatModel

class TestCatModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCatModel, self).__init__(*args, **kwargs)
        self.catModel = CatModel()

    def test_sigmoid(self) -> None:
        "Test sigmoid output range to be within 0-1"
        inputs = [-100,0,100]
        for i in inputs:
            output = self.catModel.sigmoid(i)
            self.assertTrue(output >= 0 and output <= 1, "Sigmoid should return a number between 0 and 1")

if __name__ == '__main__':
    unittest.main()