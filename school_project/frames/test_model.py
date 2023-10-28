import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk

class TestMNISTFrame(tk.Frame): #TODO
    """Frame for Cat Recognition page."""
    def __init__(self, root: tk.Tk, width: int, height: int, use_gpu: bool, model) -> None:
        """Initialise image recognition frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup image recognition frame variables
        self.use_gpu: bool = use_gpu
        self.model = model
        
        # Setup widgets
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.image_figure: Figure = Figure()
        self.image_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.image_figure,
                                                    master=self
                                                    )
        
        # Pack widgets
        self.model_status_label.pack()
        
        # Setup frame attributes
        self.pack_propagate(False)

        # Start predicting thread
        self.model_status_label.configure(
                        text="Testing trained model",
                        fg='green'
                        )
        self.test_thread: threading.Thread = threading.Thread(
                                    target=self.model.predict
                                    )
        self.test_thread.start()

    def plot_results(self):
        if not self.use_gpu:

            # Output example prediction results
            test_prediction = np.squeeze(self.model.test_prediction).T.tolist()
            test_inputs = np.squeeze(self.model.test_inputs).T
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(100 - np.mean(np.abs(self.model.test_prediction.round() - self.model.test_outputs)) * 100)}%\n" +
             f"Network Shape: " +
             f"{','.join(self.model.layers_shape)}\n"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            image1.set_title(test_prediction[0].index(max(test_prediction[0])))
            image1.imshow(test_inputs[0].reshape((28,28)))

            image2: Figure.axes = self.image_figure.add_subplot(122)
            image2.set_title(test_prediction[14].index(max(test_prediction[14])))
            image2.imshow(test_inputs[14].reshape((28,28)))

            self.image_canvas.get_tk_widget().pack(side='right')
            
            self.train_button['state'] = 'normal'

        elif self.use_gpu:

            import cupy as cp
            
            # Output example prediction results
            test_prediction = cp.squeeze(self.model.test_prediction).T.tolist()
            test_inputs = cp.asnumpy(cp.squeeze(self.model.test_inputs)).T
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(100 - np.mean(np.abs(cp.asnumpy(self.model.test_prediction).round() - cp.asnumpy(self.model.test_outputs))) * 100)}%\n" +
             f"Network Shape: " +
             f"{','.join(self.model.layers_shape)}\n"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            image1.set_title(test_prediction[0].index(max(test_prediction[0])))
            image1.imshow(test_inputs[0].reshape((28,28)))

            image2: Figure.axes = self.image_figure.add_subplot(122)
            image2.set_title(test_prediction[14].index(max(test_prediction[14])))
            image2.imshow(test_inputs[14].reshape((28,28)))

            self.image_canvas.get_tk_widget().pack(side='right')

class TestCatRecognitionFrame(tk.Frame):
    """Frame for Cat Recognition page."""
    def __init__(self, root: tk.Tk, width: int, height: int, use_gpu: bool, model) -> None:
        """Initialise image recognition frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup image recognition frame variables
        self.use_gpu: bool = use_gpu
        self.model = model
        
        # Setup widgets
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.image_figure: Figure = Figure()
        self.image_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.image_figure,
                                                    master=self
                                                    )
        
        # Pack widgets
        self.model_status_label.pack()
        
        # Setup frame attributes
        self.pack_propagate(False)

        # Start predicting thread
        self.model_status_label.configure(
                        text="Testing trained model",
                        fg='green'
                        )
        self.test_thread: threading.Thread = threading.Thread(
                                    target=self.model.predict
                                    )
        self.test_thread.start()

    def plot_results(self):
        if not self.use_gpu:
            
            # Output example prediction results
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(100 - np.mean(np.abs(self.model.test_prediction.round() - self.model.test_outputs)) * 100)}%\n" +
             f"Network Shape: " +
             f"{','.join(self.model.layers_shape)}\n"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            if np.squeeze(self.model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(
                     self.model.test_inputs[:,0].reshape((64,64,3))
                     )
            image2: Figure.axes = self.image_figure.add_subplot(122)
            if np.squeeze(self.model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(
                    self.model.test_inputs[:,14].reshape((64,64,3))
                    )
            self.image_canvas.get_tk_widget().pack(side='right')
            
            self.train_button['state'] = 'normal'
        
        elif self.use_gpu:

            import cupy as cp
            
            # Output example prediction results
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(100 - np.mean(np.abs(cp.asnumpy(self.model.test_prediction).round() - cp.asnumpy(self.model.test_outputs))) * 100)}%\n" +
             f"Network Shape: " +
             f"{','.join(self.model.layers_shape)}\n"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            if cp.squeeze(self.model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(
                     cp.asnumpy(self.model.test_inputs)[:,0].reshape((64,64,3))
                     )
            image2: Figure.axes = self.image_figure.add_subplot(122)
            if cp.squeeze(self.model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(
                    cp.asnumpy(self.model.test_inputs)[:,14].reshape((64,64,3))
                    )
            self.image_canvas.get_tk_widget().pack(side='right')

class TestXORFrame(tk.Frame): #TODO
    """Frame for Cat Recognition page."""
    def __init__(self, root: tk.Tk, width: int, height: int, model) -> None:
        """Initialise image recognition frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup image recognition frame variables
        self.model = model
        
        # Setup widgets
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.results_label: tk.Label = tk.Label(master=self,
                                                bg='white',
                                                font=('Arial', 20))
        
        # Pack widgets
        self.model_status_label.pack()
        
        # Setup frame attributes
        self.pack_propagate(False)

        # Start predicting thread
        self.model_status_label.configure(
                        text="Testing trained model",
                        fg='green'
                        )
        self.test_thread: threading.Thread = threading.Thread(
                                    target=self.model.predict
                                    )
        self.test_thread.start()

    def plot_results(self):
        # Output example prediction results
        results: str = (
                    f"Prediction Accuracy: " +
                    f"{round(self.model.test_prediction_accuracy)}%\n" +
                    f"Network Shape: " +
                    f"{','.join(self.model.layers_shape)}\n"
                    )
        for i in range(self.model.test_inputs.shape[1]):
            results += f"{self.model.test_inputs[0][i]},"
            results += f"{self.model.test_inputs[1][i]} = "
            if np.squeeze(self.model.test_prediction)[i] >= 0.5:
                results += "1\n"
            else:
                results += "0\n"
        self.results_label.configure(text=results)
        self.results_label.pack()