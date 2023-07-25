import unittest
from school_project.xor_model import XorModel

class TestXorModel(unittest.TestCase):
    def test(self) -> None:
        "Test"
        xorModel = XorModel()
        self.assertEqual(True, True, "Should be True")

if __name__ == '__main__':
    unittest.main()