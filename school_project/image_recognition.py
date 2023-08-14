import os
import threading

import h5py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import tkinter.font as tkf

class CatModel():
    """ANN model that trains to predict if an image is a cat or not a cat."""
    def __init__(self) -> None:
        """Initialise model values."""
        
        # Setup model data
        self.train_inputs, self.train_outputs,\
        self.test_inputs, self.test_outputs = self.load_datasets()
        self.train_losses: list[float] = []
        self.test_prediction: np.ndarray = None
        self.test_prediction_accuracy: float = None

        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.output_neuron_count: int = 1
        
        # Initialise weights and bias to 0/s
        self.weights: np.ndarray = np.zeros(shape=(self.input_neuron_count,
                                                   self.output_neuron_count))
        self.bias: float = 0
        self.LEARNING_RATE: float = 0.001

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Weights: {self.weights}\n" +
                f"Bias: {self.bias}\n" +
                f"Learning Rate: {self.LEARNING_RATE}")
    
    def init_model_values(self) -> None:
        """Initialise weights and bias to 0/s."""
        self.weights = np.zeros(shape=(self.train_inputs.shape[0], 1))
        self.bias = 0
    
    def load_datasets(self) -> tuple[np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray]:
        """Load image datasets.
        
        Returns:
            image input and output arrays for training and testing.
        Raises:
            FileNotFoundError: if file does not exist.

        """
        
        # Load datasets from h5 files
        # (h5 files stores large amount of data with quick access)
        train_dataset: h5py.File = h5py.File(
                                      r'school_project/datasets/train-cat.h5',
                                      'r'
                                      )
        test_dataset: h5py.File = h5py.File(
                                      r'school_project/datasets/test-cat.h5',
                                      'r'
                                      )
        
        # Load input arrays,
        # containing the RGB values for each pixel in each 64x64 pixel image,
        # for 209 images
        train_inputs: np.ndarray = np.array(train_dataset['train_set_x'][:])
        test_inputs: np.ndarray = np.array(test_dataset['test_set_x'][:])
        
        # Load output arrays of 1s for cat and 0s for not cat
        train_outputs: np.ndarray = np.array(train_dataset['train_set_y'][:])
        test_outputs: np.ndarray = np.array(test_dataset['test_set_y'][:])
        
        # Reshape input arrays into 1 dimension (flatten),
        # then divide by 255 (RGB)
        # to standardize them to a number between 0 and 1
        train_inputs = train_inputs.reshape((train_inputs.shape[0], -1)).T / 255
        test_inputs = test_inputs.reshape((test_inputs.shape[0], -1)).T / 255
        
        # Reshape output arrays into a 1 dimensional list of outputs
        train_outputs = train_outputs.reshape((1, train_outputs.shape[0]))
        test_outputs = test_outputs.reshape((1, test_outputs.shape[0]))
        return train_inputs, train_outputs, test_inputs, test_outputs

    def sigmoid(self, z: np.ndarray | int | float) -> np.ndarray | float:
        """Transfer function, transform input to number between 0 and 1.

        Args:
            z (numpy.ndarray | int | float):
            the numpy.ndarray | int | float to be transferred.
        Returns:
            numpy.ndarray | float,
            with all values | the value transferred to a number between 0-1.
        Raises:
            TypeError: if z is not of type numpy.ndarray | int | float.

        """
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, prediction: np.ndarray) -> None:
        """Adjust the weights and bias via gradient descent.
        
        Args:
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            ValueError:
            if prediction is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        weight_gradient: np.ndarray = np.dot(self.train_inputs, (prediction - self.train_outputs).T) / self.train_inputs.shape[1]
        bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.train_inputs.shape[1]
        
        # Update weights and bias
        self.weights -= self.LEARNING_RATE * weight_gradient
        self.bias -= self.LEARNING_RATE * bias_gradient

    def forward_propagation(self) -> np.ndarray:
        """Generate a prediction with the weights and bias.
        
        Returns:
            numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.weights.T, self.train_inputs) + self.bias
        prediction: np.ndarray = self.sigmoid(z1)
        return prediction

    def predict(self) -> None:
        """Use trained weights and bias
           to predict if image is a cat or not a cat."""
        
        # Calculate prediction for test dataset
        z1: np.ndarray = np.dot(self.weights.T, self.test_inputs) + self.bias
        self.test_prediction = self.sigmoid(z1)
        
        # Calculate performance of model
        self.test_prediction_accuracy = 100 - np.mean(
                                              np.abs(
                                                  self.test_prediction.round()
                                                  - self.test_outputs
                                                  )
                                              ) * 100

    def train(self, epochs: int) -> None:
        """Train weights and bias.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        
        """
        self.train_losses = []
        for epoch in range(epochs):
            if not self.running:
                break
            prediction = self.forward_propagation()
            loss: float = - (1/self.train_inputs.shape[1]) * np.sum(self.train_outputs * np.log(prediction) + (1 - self.train_outputs) * np.log(1 - prediction))
            self.train_losses.append(np.squeeze(loss))
            self.back_propagation(prediction=prediction)

class ImageRecognitionFrame(tk.Frame):
    """Frame for image recognition page."""
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
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
        self.cat_model: CatModel = CatModel()
        
        # Setup widgets
        self.menu_frame: tk.Frame = tk.Frame(master=self, bg='white')
        self.title_label: tk.Label = tk.Label(master=self.menu_frame,
                                              bg='white',
                                              font=('Arial', 20),
                                              text="Image Recognition")
        self.about_label: tk.Label = tk.Label(
                                     master=self.menu_frame,
                                     bg='white',
                                     font=('Arial', 14),
                                     text="An Image model trained on " +
                                     "recognising if an image is a cat or not"
                                    )
        self.theory_button: tk.Button = tk.Button(
                                          master=self.menu_frame, 
                                          width=13,
                                          height=1,
                                          font=tkf.Font(size=12),
                                          text="View Theory")
        if os.name == 'posix':
            self.theory_button.configure(command=lambda: os.system(
                                                  r'open docs/image_model.pdf'
                                                  ))
        elif os.name == 'nt':
            self.theory_button.configure(command=lambda: os.system(
                                                       r'.\docs\image_model.pdf'
                                                       ))
        self.train_button: tk.Button = tk.Button(master=self.menu_frame,
                                                 width=13, height=1,
                                                 font=tkf.Font(size=12),
                                                 text="Train Model",
                                                 command=self.start_training)
        self.learning_rate_scale: tk.Scale = tk.Scale(master=self.menu_frame,
                                                      bg='white',
                                                      orient='horizontal',
                                                      label="Learning Rate",
                                                      length=185,
                                                      from_=0,
                                                      to=0.037,
                                                      resolution=0.001)
        self.learning_rate_scale.set(value=self.cat_model.LEARNING_RATE)
        self.model_status_label: tk.Label = tk.Label(master=self.menu_frame,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.results_frame: tk.Frame = tk.Frame(master=self, bg='white')
        self.loss_figure: Figure = Figure()
        self.loss_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.loss_figure,
                                                    master=self.results_frame
                                                    )
        self.image_figure: Figure = Figure()
        self.image_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.image_figure,
                                                    master=self.results_frame
                                                    )
        
        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=2)
        self.about_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        self.theory_button.grid(row=2, column=0, pady=(10, 0))
        self.train_button.grid(row=2, column=1, pady=(10, 0))
        self.learning_rate_scale.grid(row=3, column=0,
                                      columnspan=2, pady=(10, 0))
        self.model_status_label.grid(row=4, column=0,
                                     columnspan=2, pady=(10, 0))
        self.menu_frame.pack()
        self.results_frame.pack(pady=(50,0))
        
        # Setup frame attributes
        self.grid_propagate(False)
        self.pack_propagate(False)

    def manage_predicting(self, predict_thread: threading.Thread) -> None:
        """Wait for model predicting thread to finish,
           then output prediction results.
        
        Args:
            predict_thread (threading.Thread):
            the thread running the image model's predict() method.
        Raises:
            TypeError: if predict_thread is not of type threading.Thread.
        
        """
        if not predict_thread.is_alive():
            
            # Output example prediction results
            self.image_figure.suptitle(
             "Prediction Correctness: " +
             f"{round(self.cat_model.test_prediction_accuracy)}%"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            if np.squeeze(self.cat_model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(self.cat_model.test_inputs[:,0].reshape((64,64,3)))
            image2: Figure.axes = self.image_figure.add_subplot(122)
            if np.squeeze(self.cat_model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(self.cat_model.test_inputs[:,14].reshape((64,64,3)))
            self.image_canvas.get_tk_widget().pack(side='right')
            
            self.train_button['state'] = 'normal'
        else:
            self.after(1_000, self.manage_predicting, predict_thread)

    def manage_training(self, train_thread: threading.Thread) -> None:
        """Wait for model training thread to finish,
           then start predicting with model in new thread.
        
        Args:
            train_thread (threading.Thread):
            the thread running the image model's train() method.
        Raises:
            TypeError: if train_thread is not of type threading.Thread.

        """
        if not train_thread.is_alive():
            
            # Plot losses of model training
            graph: Figure.axes = self.loss_figure.add_subplot(111)
            graph.set_title(f"Learning rate: {self.cat_model.LEARNING_RATE}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.cat_model.train_losses))
            self.loss_canvas.get_tk_widget().pack(side='left')
            
            # Start predicting thread
            self.model_status_label.configure(
                             text="Using trained weights and bias to predict",
                             fg='green'
                             )
            predict_thread: threading.Thread = threading.Thread(
                                                target=self.cat_model.predict
                                                )
            predict_thread.start()
            self.manage_predicting(predict_thread=predict_thread)
        else:
            self.after(1_000, self.manage_training, train_thread)

    def start_training(self) -> None:
        """Start training model in new thread."""
        self.train_button['state'] = 'disabled'
        
        # Reset canvases and figures
        self.loss_figure = Figure()
        self.loss_canvas.get_tk_widget().destroy()
        self.loss_canvas = FigureCanvasTkAgg(figure=self.loss_figure,
                                             master=self.results_frame)
        self.image_figure = Figure()
        self.image_canvas.get_tk_widget().destroy()
        self.image_canvas = FigureCanvasTkAgg(figure=self.image_figure,
                                              master=self.results_frame)
        
        # Start training thread
        self.cat_model.LEARNING_RATE = self.learning_rate_scale.get()
        self.cat_model.init_model_values()
        self.model_status_label.configure(text="Training weights and bias...",
                                          fg='red')
        train_thread: threading.Thread = threading.Thread(
                                                  target=self.cat_model.train,
                                                  args=(5_000,)
                                                  )
        train_thread.start()
        self.manage_training(train_thread=train_thread)