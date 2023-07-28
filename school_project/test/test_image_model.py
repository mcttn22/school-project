import unittest
from school_project.image_model import ImageModel

class TestImageModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageModel, self).__init__(*args, **kwargs)
        self.imageModel = ImageModel()

    def test(self) -> None:
        "Test"
        self.assertEqual(True, True, "Test")

if __name__ == '__main__':
    unittest.main()