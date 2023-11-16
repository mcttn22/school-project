import json
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import tkinter.font as tkf

class HyperParameterFrame(tk.Frame):
    """Frame for hyper-parameter page."""
    def __init__(self, root: tk.Tk, width: int, 
                 height: int, bg: str, dataset: str) -> None:
        """Initialise hyper-parameter frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            dataset (str): the name of the dataset to use
            ('MNIST', 'Cat Recognition' or 'XOR')
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg=bg)
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        self.BG = bg
        
        # Setup hyper-parameter frame variables
        self.dataset = dataset
        self.use_gpu: bool
        self.default_hyper_parameters = self.load_default_hyper_parameters(
                                                               dataset=dataset
                                                               )
        
        # Setup widgets
        self.title_label = tk.Label(master=self,
                                    bg=self.BG,
                                    font=('Arial', 20),
                                    text=dataset)
        self.about_label = tk.Label(
                             master=self,
                             bg=self.BG,
                             font=('Arial', 14),
                             text=self.default_hyper_parameters['description']
                             )
        self.learning_rate_scale = tk.Scale(
                          master=self,
                          bg=self.BG,
                          orient='horizontal',
                          label="Learning Rate",
                          length=185,
                          from_=0,
                          to=self.default_hyper_parameters['maxLearningRate'],
                          resolution=0.01
                          )
        self.learning_rate_scale.set(value=0.1)
        self.epoch_count_scale = tk.Scale(master=self,
                                          bg=self.BG,
                                          orient='horizontal',
                                          label="Epoch Count",
                                          length=185,
                                          from_=0,
                                          to=10_000,
                                          resolution=100)
        self.epoch_count_scale.set(
                             value=self.default_hyper_parameters['epochCount']
                             )
        self.train_dataset_size_scale = tk.Scale(
                   master=self,
                   bg=self.BG,
                   orient='horizontal',
                   label="Train Dataset Size",
                   length=185,
                   from_=self.default_hyper_parameters['minTrainDatasetSize'],
                   to=self.default_hyper_parameters['maxTrainDatasetSize'],
                   resolution=1
                   )
        self.train_dataset_size_scale.set(
                    value=self.default_hyper_parameters['maxTrainDatasetSize']
                    )
        self.hidden_layers_shape_label = tk.Label(
                                master=self,
                                bg=self.BG,
                                font=('Arial', 12),
                                text="Enter the number of neurons in each\n" +
                                        "hidden layer, separated by commas:"
                                )
        self.hidden_layers_shape_entry = tk.Entry(master=self)
        self.hidden_layers_shape_entry.insert(0, ",".join(
            f"{neuron_count}" for neuron_count in self.default_hyper_parameters['hiddenLayersShape']
            ))
        self.use_relu_check_button_var = tk.BooleanVar(value=True)
        self.use_relu_check_button = tk.Checkbutton(
                                        master=self,
                                        width=13, height=1,
                                        font=tkf.Font(size=12),
                                        text="Use ReLu",
                                        variable=self.use_relu_check_button_var
                                                       )
        self.use_gpu_check_button_var = tk.BooleanVar()
        self.use_gpu_check_button = tk.Checkbutton(
                                        master=self,
                                        width=13, height=1,
                                        font=tkf.Font(size=12),
                                        text="Use GPU",
                                        variable=self.use_gpu_check_button_var
                                                       )
        self.model_status_label = tk.Label(master=self,
                                           bg=self.BG,
                                           font=('Arial', 15))
        
        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=3)
        self.about_label.grid(row=1, column=0, columnspan=3)
        self.learning_rate_scale.grid(row=2, column=0, pady=(50,0))
        self.epoch_count_scale.grid(row=3, column=0, pady=(30,0))
        self.train_dataset_size_scale.grid(row=4, column=0, pady=(30,0))
        self.hidden_layers_shape_label.grid(row=2, column=1,
                                            padx=30, pady=(50,0))
        self.hidden_layers_shape_entry.grid(row=3, column=1, padx=30)
        self.use_relu_check_button.grid(row=2, column=2, pady=(30, 0))
        self.use_gpu_check_button.grid(row=3, column=2, pady=(30, 0))
        self.model_status_label.grid(row=5, column=0,
                                     columnspan=3, pady=50)
        
    def load_default_hyper_parameters(self, dataset: str) -> dict[
                                                 str, 
                                                 str | int | list[int] | float
                                                 ]:
        """Load the dataset's default hyper-parameters from the json file.
           
           Args:
               dataset (str): the name of the dataset to load hyper-parameters
               for. ('MNIST', 'Cat Recognition' or 'XOR')
            Returns:
                a dictionary of default hyper-parameter values.
        """
        with open('school_project/frames/hyper-parameter-defaults.json') as f:
            return json.load(f)[dataset]
    
    def create_model(self) -> object:
        """Create and return a Model using the hyper-parameters set.

           Returns:
               a Model object.
        """
        self.use_gpu = self.use_gpu_check_button_var.get()

        # Validate hidden layers shape input
        hidden_layers_shape_input = [layer for layer in self.hidden_layers_shape_entry.get().replace(' ', '').split(',') if layer != '']
        for layer in hidden_layers_shape_input:
            if not layer.isdigit():
                self.model_status_label.configure(
                                        text="Invalid hidden layers shape",
                                        fg='red'
                                        )
                raise ValueError

        # Create Model
        if not self.use_gpu:
            if self.dataset == "MNIST":
                from school_project.models.cpu.mnist import MNISTModel as Model
            elif self.dataset == "Cat Recognition":
                from school_project.models.cpu.cat_recognition import CatRecognitionModel as Model
            elif self.dataset == "XOR":
                from school_project.models.cpu.xor import XORModel as Model
            model = Model(hidden_layers_shape = [int(neuron_count) for neuron_count in hidden_layers_shape_input],
                          train_dataset_size = self.train_dataset_size_scale.get(),
                          learning_rate = self.learning_rate_scale.get(),
                          use_relu = self.use_relu_check_button_var.get())
            model.init_random_values()

        else:
            try:
                if self.dataset == "MNIST":
                    from school_project.models.gpu.mnist import MNISTModel as Model
                elif self.dataset == "Cat Recognition":
                    from school_project.models.gpu.cat_recognition import CatRecognitionModel as Model
                elif self.dataset == "XOR":
                    from school_project.models.gpu.xor import XORModel as Model
                model = Model(hidden_layers_shape = [int(neuron_count) for neuron_count in hidden_layers_shape_input],
                              train_dataset_size = self.train_dataset_size_scale.get(),
                              learning_rate = self.learning_rate_scale.get(),
                              use_relu = self.use_relu_check_button_var.get())
                model.init_random_values()
            except ImportError as ie:
                self.model_status_label.configure(
                                        text="Failed to initialise GPU",
                                        fg='red'
                                        )
                raise ImportError
        return model
        
class TrainingFrame(tk.Frame):
    """Frame for training page."""
    def __init__(self, root: tk.Tk, width: int,
                 height: int, bg: str,
                 model: object, epoch_count: int) -> None:
        """Initialise training frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            model (object): the Model object to be trained.
            epoch_count (int): the number of training epochs.
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
        self.training_progress_label = tk.Label(master=self,
                                                bg=self.BG,
                                                font=('Arial', 15))
        self.loss_figure: Figure = Figure()
        self.loss_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                      figure=self.loss_figure,
                                                      master=self
                                                      )
        
        # Pack widgets
        self.model_status_label.pack(pady=(30,0))
        self.training_progress_label.pack(pady=30)
        
        # Start training thread
        self.model_status_label.configure(
                                        text="Training weights and biases...",
                                        fg='red'
                                        )
        self.train_thread: threading.Thread = threading.Thread(
                                                           target=model.train,
                                                           args=(epoch_count,)
                                                           )
        self.train_thread.start()

    def plot_losses(self, model: object) -> None:
        """Plot losses of Model training.
        
           Args:
               model (object): the Model object thats been trained.
        
        """
        self.model_status_label.configure(
                 text=f"Weights and biases trained in {model.training_time}s",
                 fg='green'
                 )
        graph: Figure.axes = self.loss_figure.add_subplot(111)
        graph.set_title("Learning rate: " +
                        f"{model.learning_rate}")
        graph.set_xlabel("Epochs")
        graph.set_ylabel("Loss Value")
        graph.plot(np.squeeze(model.train_losses))
        self.loss_canvas.get_tk_widget().pack()