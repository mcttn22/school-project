import os
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import tkinter.font as tkf

class XorModel():
    """ANN model
      that trains to predict the output of a XOR gate with two inputs."""
    def __init__(self) -> None:
        """Initialise model values."""

        # Setup model data
        self.train_inputs: np.ndarray = np.array([[0, 0, 1, 1],
                                                  [0, 1, 0, 1]])
        self.train_outputs: np.ndarray = np.array([[0, 1, 1, 0]])
        self.train_losses: list[float]
        self.test_prediction: np.ndarray
        self.test_prediction_accuracy: float
        
        # Setup model attributes
        self.running: bool = True
        self.input_neuron_count: int = self.train_inputs.shape[0]
        self.hidden_neuron_count: int = 2
        self.output_neuron_count: int = 1
        
        # Setup weights and biases
        np.random.seed(2)  # Sets up pseudo random values for weight arrays
        self.hidden_weights: np.ndarray
        self.output_weights: np.ndarray
        self.hidden_biases: np.ndarray
        self.output_biases: np.ndarray
        self.LEARNING_RATE: float = 0.1

    def __repr__(self) -> str:
        """Read current state of model.
        
        Returns:
            a string description of the model's weights,
            bias and learning rate values.

        """
        return (f"Number of hidden neurons: {self.hidden_neuron_count}\n" +
                f"Hidden Weights: {self.hidden_weights.tolist()}\n" +
                f"Output Weights: {self.output_weights.tolist()}\n" +
                f"Hidden biases: {self.hidden_biases.tolist()}\n" +
                f"Output biases: {self.output_biases.tolist()}\n" +
                f"Learning Rate: {self.LEARNING_RATE}")

    def init_model_values(self) -> None:
        """Initialise weights to randdom values and biases to 0s"""
        self.hidden_weights = np.random.rand(self.hidden_neuron_count,
                                             self.input_neuron_count)
        self.output_weights = np.random.rand(self.output_neuron_count,
                                             self.hidden_neuron_count)
        self.hidden_biases: np.ndarray = np.zeros(
                                          shape=(self.hidden_neuron_count, 1)
                                          )
        self.output_biases: np.ndarray = np.zeros(
                                          shape=(self.output_neuron_count, 1))

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

    def back_propagation(self, hidden_output: np.ndarray,
                         prediction: np.ndarray) -> None:
        """Adjust the weights and biases via gradient descent.
        
        Args:
            hidden_output (numpy.ndarray): the matrice of hidden output values
            prediction (numpy.ndarray): the matrice of prediction values
        Raises:
            ValueError:
            if prediction or hidden_output
            is not a suitable multiplier with the weights
            (incorrect shape)
        
        """
        output_weight_gradient: np.ndarray = np.dot(prediction - self.train_outputs, hidden_output.T) / self.train_inputs.shape[1]
        hidden_weight_gradient: np.ndarray = np.dot(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output), self.train_inputs.T) / self.train_inputs.shape[1]
        output_bias_gradient: np.ndarray = np.sum(prediction - self.train_outputs) / self.train_inputs.shape[1]
        hidden_bias_gradient: np.ndarray = np.sum(np.dot(self.output_weights.T, prediction - self.train_outputs) * hidden_output * (1 - hidden_output)) / self.train_inputs.shape[1]

        # Reshape arrays to match the weight arrays for multiplication
        output_weight_gradient = np.reshape(output_weight_gradient,
                                            self.output_weights.shape)
        hidden_weight_gradient = np.reshape(hidden_weight_gradient,
                                            self.hidden_weights.shape)
        
        # Update weights and biases
        self.output_weights -= self.LEARNING_RATE * output_weight_gradient
        self.hidden_weights -= self.LEARNING_RATE * hidden_weight_gradient
        self.output_biases -= self.LEARNING_RATE * output_bias_gradient
        self.hidden_biases -= self.LEARNING_RATE * hidden_bias_gradient

    def forward_propagation(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a prediction with the weights and biases.
        
        Returns:
            a numpy.ndarray of the hidden output values
            and a numpy.ndarray of prediction values.

        """
        z1: np.ndarray = np.dot(self.hidden_weights, self.train_inputs) + self.hidden_biases
        hidden_output: np.ndarray = self.sigmoid(z1)
        z2: np.ndarray = np.dot(self.output_weights, hidden_output) + self.output_biases
        prediction: np.ndarray = self.sigmoid(z2)
        return hidden_output, prediction

    def predict(self) -> None:
        """Use trained weights and biases to predict ouput of XOR gate on two inputs."""
        hidden_output, self.test_prediction = self.forward_propagation()
        
        # Calculate performance of model
        self.test_prediction_accuracy = 100 - np.mean(
                                              np.abs(self.test_prediction
                                                     - self.train_outputs)
                                              ) * 100

    def train(self, epochs: int) -> None:
        """Train weights and biases.
        
        Args:
            epochs (int): the number of forward and back propagations to do.
        
        """
        self.train_losses = []
        for epoch in range(epochs):
            if not self.running:
                break
            hidden_output, prediction = self.forward_propagation()
            loss: float = - (1/self.train_inputs.shape[1]) * np.sum(self.train_outputs * np.log(prediction) + (1 - self.train_outputs) * np.log(1 - prediction))
            self.train_losses.append(loss)
            self.back_propagation(hidden_output=hidden_output,
                                  prediction=prediction)

class ExperimentsFrame(tk.Frame):
    """Frame for experiments page."""
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        """Initialise experiments frame widgets.
        
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
        
        # Setup experiments frame variables
        self.xor_model = XorModel()
        
        # Setup widgets
        self.menu_frame: tk.Frame = tk.Frame(master=self, bg='white')
        self.title_label: tk.Label = tk.Label(master=self.menu_frame,
                                              bg='white',
                                              font=('Arial', 20),
                                              text="Experiments")
        self.about_label: tk.Label = tk.Label(
         master=self.menu_frame, 
         bg='white', 
         font=('Arial', 14), 
         text="For experimenting with Artificial Neural Networks, " +
              "a XOR gate model has been used for its lesser computation time"
         )
        self.theory_button: tk.Button = tk.Button(master=self.menu_frame,
                                                  width=13,
                                                  height=1,
                                                  font=tkf.Font(size=12),
                                                  text="View Theory")
        if os.name == 'posix':
            self.theory_button.configure(command=lambda: os.system(
                                                    r'open docs/xor_model.pdf'
                                                    ))
        elif os.name == 'nt':
            self.theory_button.configure(command=lambda: os.system(
                                                         r'.\docs\xor_model.pdf'
                                                         ))
        self.train_button: tk.Button = tk.Button(master=self.menu_frame,
                                                 width=13,
                                                 height=1,
                                                 font=tkf.Font(size=12),
                                                 text="Train Model",
                                                 command=self.start_training)
        self.learning_rate_scale: tk.Scale = tk.Scale(master=self.menu_frame,
                                                      bg='white',
                                                      orient='horizontal',
                                                      label="Learning Rate",
                                                      length=185,
                                                      from_=0,
                                                      to=1,
                                                      resolution=0.01)
        self.learning_rate_scale.set(value=self.xor_model.LEARNING_RATE)
        self.hidden_neuron_count_scale: tk.Scale = tk.Scale(
                                             master=self.menu_frame,
                                             bg='white',
                                             orient='horizontal',
                                             label="Number of Hidden Neurons",
                                             length=185,
                                             from_=1,
                                             to=20,
                                             resolution=1.0
                                             )
        self.hidden_neuron_count_scale.set(
                                     value=self.xor_model.hidden_neuron_count
                                     )
        self.model_status_label: tk.Label = tk.Label(master=self.menu_frame,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.results_frame: tk.Frame = tk.Frame(master=self, bg='white')
        self.loss_figure: Figure = Figure()
        self.loss_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.loss_figure,
                                                    master=self.results_frame
                                                    )
        self.results_label: tk.Label = tk.Label(master=self.results_frame,
                                                bg='white',
                                                font=('Arial', 20))
        
        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=4)
        self.about_label.grid(row=1, column=0, columnspan=4, pady=(10,0))
        self.theory_button.grid(row=2, column=0, pady=(10,0))
        self.train_button.grid(row=2, column=3, pady=(10,0))
        self.learning_rate_scale.grid(row=3, column=1, padx=(0,5),
                                      pady=(10,0), sticky='e')
        self.hidden_neuron_count_scale.grid(row=3, column=2, padx=(5,0),
                                           pady=(10,0), sticky='w')
        self.model_status_label.grid(row=4, column=0, columnspan=4,
                                     pady=(10,0))
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
            results: str = (
                      f"Prediction Accuracy: " +
                      f"{round(self.xor_model.test_prediction_accuracy)}%\n" +
                      f"Number of Hidden Neurons: " +
                      f"{self.xor_model.hidden_neuron_count}\n"
                      )
            for i in range(self.xor_model.train_inputs.shape[1]):
                results += f"{self.xor_model.train_inputs[0][i]},"
                results += f"{self.xor_model.train_inputs[1][i]} = "
                if np.squeeze(self.xor_model.test_prediction)[i] >= 0.5:
                    results += "1\n"
                else:
                    results += "0\n"
            self.results_label.configure(text=results)
            self.results_label.pack(side='right')
            
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
            graph.set_title(f"Learning rate: {self.xor_model.LEARNING_RATE}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.xor_model.train_losses))
            self.loss_canvas.get_tk_widget().pack(side="left")
            
            # Start predicting thread
            self.model_status_label.configure(
                                      fg='green',
                                      text="Using trained weights and biases to predict"
                                      )
            predict_thread: threading.Thread = threading.Thread(
                                                target=self.xor_model.predict
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
        self.results_label.pack_forget()
        
        # Start training thread
        self.xor_model.LEARNING_RATE = self.learning_rate_scale.get()
        self.xor_model.hidden_neuron_count = self.hidden_neuron_count_scale.get()
        self.xor_model.init_model_values()
        self.model_status_label.configure(text="Training weights and biases...",
                                          fg='red')
        train_thread: threading.Thread = threading.Thread(
                                                  target=self.xor_model.train,
                                                  args=(50_000,)
                                                  )
        train_thread.start()
        self.manage_training(train_thread=train_thread)