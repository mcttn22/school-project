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
                 height: int, dataset: str) -> None:
        """Initialise hyper-parameter frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            dataset (str): the name of the dataset to use
            ('MNIST', 'Cat Recognition' or 'XOR')
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup hyper-parameter frame variables
        self.dataset = dataset
        self.use_gpu: bool
        self.default_hyper_parameters = self.load_default_hyper_parameters(dataset=dataset)
        
        # Setup widgets
        self.title_label: tk.Label = tk.Label(master=self,
                                              bg='white',
                                              font=('Arial', 20),
                                              text=dataset)
        self.learning_rate_scale: tk.Scale = tk.Scale(master=self,
                                                      bg='white',
                                                      orient='horizontal',
                                                      label="Learning Rate",
                                                      length=185,
                                                      from_=0,
                                                      to=self.default_hyper_parameters['maxLearningRate'],
                                                      resolution=0.01)
        self.learning_rate_scale.set(value=0.1)
        self.epoch_count_scale: tk.Scale = tk.Scale(master=self,
                                                      bg='white',
                                                      orient='horizontal',
                                                      label="Epoch Count",
                                                      length=185,
                                                      from_=0,
                                                      to=10_000,
                                                      resolution=100)
        self.epoch_count_scale.set(value=self.default_hyper_parameters['epochCount'])
        self.hidden_layers_shape_label: tk.Label = tk.Label(
                                master=self,
                                bg='white',
                                font=('Arial', 12),
                                text="Enter the number of neurons in each\n" +
                                        "hidden layer, separated by commas:"
                                )
        self.hidden_layers_shape_entry: tk.Entry = tk.Entry(master=self)
        self.hidden_layers_shape_entry.insert(0, ",".join(
                           f"{neuron_count}" for neuron_count in self.default_hyper_parameters['hiddenLayersShape']
                           ))
        self.use_gpu_check_button_var: tk.BooleanVar = tk.BooleanVar()
        self.use_gpu_check_button: tk.Checkbutton = tk.Checkbutton(
                                        master=self,
                                        width=13, height=1,
                                        font=tkf.Font(size=12),
                                        text="Use GPU",
                                        variable=self.use_gpu_check_button_var
                                                       )
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        
        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=3)
        self.learning_rate_scale.grid(row=1, column=0, pady=(50,0))
        self.epoch_count_scale.grid(row=2, column=0, pady=(30,0))
        self.hidden_layers_shape_label.grid(row=1, column=1,
                                            padx=30, pady=(50,0))
        self.hidden_layers_shape_entry.grid(row=2, column=1, padx=30)
        self.use_gpu_check_button.grid(row=2, column=2, pady=(30, 0))
        self.model_status_label.grid(row=3, column=0,
                                     columnspan=3, pady=50)
        
    def load_default_hyper_parameters(self, dataset: str) -> dict[
                                                       str, 
                                                       int | list[int] | float
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
                from school_project.models.cpu.mnist import Model
            elif self.dataset == "Cat Recognition":
                from school_project.models.cpu.cat_recognition import Model
            elif self.dataset == "XOR":
                from school_project.models.cpu.xor import Model
            model = Model(hidden_layers_shape = [int(neuron_count) for neuron_count in hidden_layers_shape_input],
                          learning_rate = self.learning_rate_scale.get(),
                          epoch_count = self.epoch_count_scale.get())

        else:
            try:
                if self.dataset == "MNIST":
                    from school_project.models.gpu.mnist import Model
                elif self.dataset == "Cat Recognition":
                    from school_project.models.gpu.cat_recognition import Model
                elif self.dataset == "XOR":
                    from school_project.models.gpu.xor import Model
                model = Model(hidden_layers_shape = [int(neuron_count) for neuron_count in hidden_layers_shape_input],
                              learning_rate = self.learning_rate_scale.get(),
                              epoch_count = self.epoch_count_scale.get())
            except ImportError as ie:
                self.model_status_label.configure(
                                        text="Failed to initialise GPU",
                                        fg='red'
                                        )
                raise ImportError
        return model
        
class TrainingFrame(tk.Frame):
    """Frame for training page."""
    def __init__(self, root: tk.Tk, width: int, height: int, model: object) -> None:
        """Initialise training frame widgets.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            model (object): the Model object to be trained.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup training frame variables
        self.model = model
        
        # Setup widgets
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        self.loss_figure: Figure = Figure()
        self.loss_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
                                                    figure=self.loss_figure,
                                                    master=self
                                                    )
        
        # Pack widgets
        self.model_status_label.pack()
        
        # Start training thread
        self.model_status_label.configure(text="Training weights and bias...",
                                          fg='red')
        self.train_thread: threading.Thread = threading.Thread(
                                                       target=self.model.train
                                                       )
        self.train_thread.start()

    def plot_losses(self) -> None:
        """Plot losses of Model training."""
        graph: Figure.axes = self.loss_figure.add_subplot(111)
        graph.set_title("Learning rate: " +
                        f"{self.model.learning_rate}")
        graph.set_xlabel("Epochs")
        graph.set_ylabel("Loss Value")
        graph.plot(np.squeeze(self.model.train_losses))
        self.loss_canvas.get_tk_widget().pack()