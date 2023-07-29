import unittest
from school_project.image_model import ImageModel

class TestImageModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageModel, self).__init__(*args, **kwargs)
        self.imageModel = ImageModel()

    def test_sigmoid(self) -> None:
        "Test sigmoid output range to be within 0-1"
        inputs = [-100,0,100]
        for i in inputs:
            output = self.imageModel.sigmoid(i)
            self.assertTrue(output >= 0 and output <= 1, "Sigmoid should return a number between 0 and 1")

if __name__ == '__main__':
    unittest.main()