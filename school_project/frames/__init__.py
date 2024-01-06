"""Package of tkinter frames for the main window."""

from .create_model import HyperParameterFrame, TrainingFrame
from .load_model import LoadModelFrame
from .test_model import TestMNISTFrame, TestCatRecognitionFrame, TestXORFrame

__all__ = ['create_model', 'load_model', 'test_model']