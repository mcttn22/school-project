import unittest
from school_project import *

class TestMain(unittest.TestCase):
    def test(self):
        self.assertEqual(True, True, "Should be True")

if __name__ == '__main__':
    unittest.main()