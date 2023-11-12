import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk

class TestMNISTFrame(tk.Frame):
    """Frame for Testing MNIST page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, use_gpu: bool, model: object) -> None:
        """Initialise test MNIST frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            use_gpu (bool): True or False whether the GPU should be used.
            model (object): The Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup test MNIST frame variables
        self.use_gpu = use_gpu
        
        # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                          bg='white',
                                          font=('Arial', 15))
        self.image_figure = Figure()
        self.image_canvas = FigureCanvasTkAgg(figure=self.image_figure,
                                              master=self)
        
        # Pack widgets
        self.model_status_label.pack()

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
            test_prediction = np.squeeze(model.test_prediction).T.tolist()
            test_inputs = np.squeeze(model.test_inputs).T
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(model.test_prediction.round() - model.test_outputs)) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )
            image1 = self.image_figure.add_subplot(121)
            image1.set_title(test_prediction[0].index(max(test_prediction[0])))
            image1.imshow(test_inputs[0].reshape((28,28)))

            image2 = self.image_figure.add_subplot(122)
            image2.set_title(test_prediction[14].index(max(test_prediction[14])))
            image2.imshow(test_inputs[14].reshape((28,28)))

            self.image_canvas.get_tk_widget().pack()

        elif self.use_gpu:

            import cupy as cp
            
            test_prediction = cp.squeeze(model.test_prediction).T.tolist()
            test_inputs = cp.asnumpy(cp.squeeze(model.test_inputs)).T
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(cp.asnumpy(model.test_prediction).round() - cp.asnumpy(model.test_outputs))) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )
            image1 = self.image_figure.add_subplot(121)
            image1.set_title(test_prediction[0].index(max(test_prediction[0])))
            image1.imshow(test_inputs[0].reshape((28,28)))

            image2 = self.image_figure.add_subplot(122)
            image2.set_title(test_prediction[14].index(max(test_prediction[14])))
            image2.imshow(test_inputs[14].reshape((28,28)))

            self.image_canvas.get_tk_widget().pack()

class TestCatRecognitionFrame(tk.Frame):
    """Frame for Testing Cat Recognition page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, use_gpu: bool, model: object) -> None:
        """Initialise test cat recognition frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            use_gpu (bool): True or False whether the GPU should be used.
            model (object): the Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup image recognition frame variables
        self.use_gpu = use_gpu
        
        # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                          bg='white',
                                          font=('Arial', 15))
        self.image_figure = Figure()
        self.image_canvas = FigureCanvasTkAgg(figure=self.image_figure,
                                              master=self)
        
        # Pack widgets
        self.model_status_label.pack(pady=(30,0))

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
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(model.test_prediction.round() - model.test_outputs)) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )
            image1 = self.image_figure.add_subplot(121)
            if np.squeeze(model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(
                     model.test_inputs[:,0].reshape((64,64,3))
                     )
            image2 = self.image_figure.add_subplot(122)
            if np.squeeze(model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(
                    model.test_inputs[:,14].reshape((64,64,3))
                    )
            self.image_canvas.get_tk_widget().pack()
        
        elif self.use_gpu:

            import cupy as cp
            
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(number=100 - np.mean(np.abs(cp.asnumpy(model.test_prediction).round() - cp.asnumpy(model.test_outputs))) * 100, ndigits=1)}%\n" +
             f"Network Shape: " +
             f"{','.join(model.layers_shape)}\n"
             )
            image1 = self.image_figure.add_subplot(121)
            if cp.squeeze(model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(
                     cp.asnumpy(model.test_inputs)[:,0].reshape((64,64,3))
                     )
            image2 = self.image_figure.add_subplot(122)
            if cp.squeeze(model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(
                    cp.asnumpy(model.test_inputs)[:,14].reshape((64,64,3))
                    )
            self.image_canvas.get_tk_widget().pack()

class TestXORFrame(tk.Frame):
    """Frame for Testing XOR page."""
    def __init__(self, root: tk.Tk, width: int, height: int, model: object) -> None:
        """Initialise test XOR frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            model (object): the Model object to be tested.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup widgets
        self.model_status_label = tk.Label(master=self,
                                           bg='white',
                                           font=('Arial', 15))
        self.results_label = tk.Label(master=self,
                                      bg='white',
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
        results: str = (
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