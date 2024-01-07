"""Tkinter frames for testing a saved Artificial Neural Network model."""

import random
import threading
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class TestMNISTFrame(tk.Frame):
    """Frame for Testing MNIST page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, bg: str,
                 use_gpu: bool, model: object) -> None:
        """Initialise test MNIST frame widgets.

        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            use_gpu (bool): True or False whether the GPU should be used.
            model (object): The Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.

        """
        super().__init__(master=root, width=width, height=height, bg=bg)
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        self.BG = bg

        # Setup test MNIST frame variables
        self.use_gpu = use_gpu

         # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                           bg=self.BG,
                                           font=('Arial', 15))
        self.results_label = tk.Label(master=self,
                                      bg=self.BG,
                                      font=('Arial', 15))
        self.correct_prediction_figure = Figure()
        self.correct_prediction_canvas = FigureCanvasTkAgg(
                                        figure=self.correct_prediction_figure,
                                        master=self
                                        )
        self.incorrect_prediction_figure = Figure()
        self.incorrect_prediction_canvas = FigureCanvasTkAgg(
                                      figure=self.incorrect_prediction_figure,
                                      master=self
                                      )

        # Grid widgets
        self.model_status_label.grid(row=0, columnspan=3, pady=(30,0))
        self.results_label.grid(row=1, columnspan=3)
        self.incorrect_prediction_canvas.get_tk_widget().grid(row=2, column=0)
        self.correct_prediction_canvas.get_tk_widget().grid(row=2, column=2)

        # Start test thread
        self.model_status_label.configure(text="Testing trained model",
                                          fg='red')
        self.test_thread = threading.Thread(target=model.test)
        self.test_thread.start()

    def plot_results(self, model: object) -> None:
        """Plot results of Model test.

           Args:
               model (object): the Model object thats been tested.

        """
        self.model_status_label.configure(text="Testing Results:", fg='green')
        if not self.use_gpu:
            self.results_label.configure(
             text="Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(model.test_prediction.round() - model.test_outputs)) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )

            test_inputs = np.squeeze(model.test_inputs).T
            test_outputs = np.squeeze(model.test_outputs).T.tolist()
            test_prediction = np.squeeze(model.test_prediction).T.tolist()

            # Randomly shuffle order of test_inputs, test_outputs and test_prediciton
            # whilst maintaining order between them
            test_data = list(zip(test_inputs,
                                 test_outputs,
                                 test_prediction))
            random.shuffle(test_data)
            test_inputs, test_outputs, test_prediction = zip(*test_data)

        elif self.use_gpu:

            import cupy as cp

            self.results_label.configure(
             text="Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(cp.asnumpy(model.test_prediction).round() - cp.asnumpy(model.test_outputs))) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )

            test_inputs = cp.asnumpy(cp.squeeze(model.test_inputs)).T
            test_outputs = cp.asnumpy(cp.squeeze(model.test_outputs)).T.tolist()
            test_prediction = cp.squeeze(model.test_prediction).T.tolist()

            # Randomly shuffle order of test_inputs, test_outputs and test_prediciton
            # whilst maintaining order between them
            test_data = list(zip(test_inputs,
                                 test_outputs,
                                 test_prediction))
            random.shuffle(test_data)
            test_inputs, test_outputs, test_prediction = zip(*test_data)

        # Setup incorrect prediction figure
        self.incorrect_prediction_figure.suptitle("Incorrect predictions:")
        image_count = 0
        for i in range(len(test_prediction)):
            if test_prediction[i].index(max(test_prediction[i])) != test_outputs[i].index(max(test_outputs[i])):
                if image_count == 2:
                    break
                elif image_count == 0:
                    image = self.incorrect_prediction_figure.add_subplot(121)
                elif image_count == 1:
                    image = self.incorrect_prediction_figure.add_subplot(122)
                image.set_title(f"Predicted: {test_prediction[i].index(max(test_prediction[i]))}\n" +
                                f"Should have predicted: {test_outputs[i].index(max(test_outputs[i]))}")
                image.imshow(test_inputs[i].reshape((28,28)))
                image_count += 1

        # Setup correct prediction figure
        self.correct_prediction_figure.suptitle("Correct predictions:")
        image_count = 0
        for i in range(len(test_prediction)):
            if test_prediction[i].index(max(test_prediction[i])) == test_outputs[i].index(max(test_outputs[i])):
                if image_count == 2:
                    break
                elif image_count == 0:
                    image = self.correct_prediction_figure.add_subplot(121)
                elif image_count == 1:
                    image = self.correct_prediction_figure.add_subplot(122)
                image.set_title(f"Predicted: {test_prediction[i].index(max(test_prediction[i]))}")
                image.imshow(test_inputs[i].reshape((28,28)))
                image_count += 1

class TestCatRecognitionFrame(tk.Frame):
    """Frame for Testing Cat Recognition page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, bg: str,
                 use_gpu: bool, model: object) -> None:
        """Initialise test cat recognition frame widgets.

        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            use_gpu (bool): True or False whether the GPU should be used.
            model (object): the Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.

        """
        super().__init__(master=root, width=width, height=height, bg=bg)
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        self.BG = bg

        # Setup image recognition frame variables
        self.use_gpu = use_gpu

        # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                           bg=self.BG,
                                           font=('Arial', 15))
        self.results_label = tk.Label(master=self,
                                      bg=self.BG,
                                      font=('Arial', 15))
        self.correct_prediction_figure = Figure()
        self.correct_prediction_canvas = FigureCanvasTkAgg(
                                        figure=self.correct_prediction_figure,
                                        master=self
                                        )
        self.incorrect_prediction_figure = Figure()
        self.incorrect_prediction_canvas = FigureCanvasTkAgg(
                                      figure=self.incorrect_prediction_figure,
                                      master=self
                                      )

        # Grid widgets
        self.model_status_label.grid(row=0, columnspan=3, pady=(30,0))
        self.results_label.grid(row=1, columnspan=3)
        self.incorrect_prediction_canvas.get_tk_widget().grid(row=2, column=0)
        self.correct_prediction_canvas.get_tk_widget().grid(row=2, column=2)

        # Start test thread
        self.model_status_label.configure(text="Testing trained model...",
                                          fg='red')
        self.test_thread = threading.Thread(target=model.test)
        self.test_thread.start()

    def plot_results(self, model: object) -> None:
        """Plot results of Model test

           Args:
               model (object): the Model object thats been tested.

        """
        self.model_status_label.configure(text="Testing Results:", fg='green')
        if not self.use_gpu:
            self.results_label.configure(
             text="Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(model.test_prediction.round() - model.test_outputs)) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )

            # Randomly shuffle order of test_inputs, test_outputs and test_prediciton
            # whilst maintaining order between them
            test_data = list(zip(model.test_inputs.T,
                                 np.squeeze(model.test_outputs).T.tolist(),
                                 np.squeeze(model.test_prediction.round()).T.tolist()))
            random.shuffle(test_data)
            (test_inputs,
             test_outputs,
             test_prediction) = map(lambda arr: np.array(arr).T,
                                    zip(*test_data))

        elif self.use_gpu:

            import cupy as cp

            self.results_label.configure(
             text="Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(cp.asnumpy(model.test_prediction).round() - cp.asnumpy(model.test_outputs))) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )

            # Randomly shuffle order of test_inputs, test_outputs and test_prediciton
            # whilst maintaining order between them
            test_data = list(zip(cp.asnumpy(model.test_inputs).T,
                                 cp.asnumpy(cp.squeeze(model.test_outputs)).T.tolist(),
                                 cp.asnumpy(cp.squeeze(model.test_prediction)).round().T.tolist()))
            random.shuffle(test_data)
            (test_inputs,
             test_outputs,
             test_prediction) = map(lambda arr: np.array(arr).T,
                                    zip(*test_data))

        # Setup incorrect prediction figure
        self.incorrect_prediction_figure.suptitle("Incorrect predictions:")
        image_count = 0
        for i in range(len(test_prediction)):
            if test_prediction[i] != test_outputs[i]:
                if image_count == 2:
                    break
                elif image_count == 0:
                    image = self.incorrect_prediction_figure.add_subplot(121)
                elif image_count == 1:
                    image = self.incorrect_prediction_figure.add_subplot(122)
                image.set_title(f"Predicted: {'Cat' if test_prediction[i] == 1 else 'Not a cat'}\n")
                image.imshow(test_inputs[:,i].reshape((64,64,3)))
                image_count += 1

        # Setup correct prediction figure
        self.correct_prediction_figure.suptitle("Correct predictions:")
        image_count = 0
        for i in range(len(test_prediction)):
            if test_prediction[i] == test_outputs[i]:
                if image_count == 2:
                    break
                elif image_count == 0:
                    image = self.correct_prediction_figure.add_subplot(121)
                elif image_count == 1:
                    image = self.correct_prediction_figure.add_subplot(122)
                image.set_title(f"Predicted: {'Cat' if test_prediction[i] == 1 else 'Not a cat'}\n")
                image.imshow(test_inputs[:,i].reshape((64,64,3)))
                image_count += 1

class TestXORFrame(tk.Frame):
    """Frame for Testing XOR page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, bg: str, model: object) -> None:
        """Initialise test XOR frame widgets.

        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            model (object): the Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.

        """
        super().__init__(master=root, width=width, height=height, bg=bg)
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        self.BG = bg

        # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                           bg=self.BG,
                                           font=('Arial', 15))
        self.results_label = tk.Label(master=self,
                                      bg=self.BG,
                                      font=('Arial', 20))

        # Pack widgets
        self.model_status_label.pack(pady=(30,0))

        # Start test thread
        self.model_status_label.configure(text="Testing trained model...",
                                          fg='red')
        self.test_thread = threading.Thread(target=model.test)
        self.test_thread.start()

    def plot_results(self, model: object):
        """Plot results of Model test.

           Args:
               model (object): the Model object thats been tested.

        """
        self.model_status_label.configure(text="Testing Results:", fg='green')
        results = (
             f"Prediction Accuracy: " +
             f"{round(number=model.test_prediction_accuracy, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )
        for i in range(model.test_inputs.shape[1]):
            results += f"{model.test_inputs[0][i]},"
            results += f"{model.test_inputs[1][i]} = "
            if np.squeeze(model.test_prediction)[i] >= 0.5:
                results += "1\n"
            else:
                results += "0\n"
        self.results_label.configure(text=results)
        self.results_label.pack()
