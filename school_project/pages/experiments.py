import os
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import tkinter.font as tkf

from school_project.models.xor import DeepModel
            
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
        self.deep_model: DeepModel = DeepModel()
        
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
        self.model_theory_button: tk.Button = tk.Button(
                                                       master=self.menu_frame,
                                                       width=15,
                                                       height=1,
                                                       font=tkf.Font(size=12),
                                                       text="View Model Theory"
                                                       )
        if os.name == 'posix':
            self.model_theory_button.configure(command=lambda: os.system(
                                   r'open docs/models/utils/deep_model.pdf'
                                   ))
        elif os.name == 'nt':
            self.model_theory_button.configure(command=lambda: os.system(
                                      r'.\docs\models\utils\deep_model.pdf'
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
        self.learning_rate_scale.set(value=self.deep_model.learning_rate)
        self.hidden_layers_shape_label: tk.Label = tk.Label(master=self.menu_frame,
                                                            bg='white',
                                                            font=('Arial', 12),
                                                            text="Enter the number of neurons in each\n" +
                                                                  "hidden layer, separated by commas:")
        self.hidden_layers_shape_entry: tk.Entry = tk.Entry(master=self.menu_frame,)
        self.hidden_layers_shape_entry.insert(0, ",".join(f"{i}" for i in self.deep_model.hidden_layers_shape))
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
        self.model_theory_button.grid(row=2, column=0, pady=(10,0))
        self.train_button.grid(row=2, column=3, pady=(10,0))
        self.hidden_layers_shape_label.grid(row=3, column=2, padx=(5,0),
                                           pady=(30,0))
        self.learning_rate_scale.grid(row=4, column=1, 
                                      padx=(0,5))
        self.hidden_layers_shape_entry.grid(row=4, column=2, 
                                            padx=(5,0))
        self.model_status_label.grid(row=5, column=0, columnspan=4,
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
            the thread running the model's predict() method.
        Raises:
            TypeError: if predict_thread is not of type threading.Thread.
        
        """
        if not predict_thread.is_alive():
            
            # Output example prediction results
            results: str = (
                      f"Prediction Accuracy: " +
                      f"{round(self.deep_model.test_prediction_accuracy)}%\n" +
                      f"Network Shape: " +
                      f"{','.join(self.deep_model.layers_shape)}\n"
                      )
            for i in range(self.deep_model.train_inputs.shape[1]):
                results += f"{self.deep_model.train_inputs[0][i]},"
                results += f"{self.deep_model.train_inputs[1][i]} = "
                if np.squeeze(self.deep_model.test_prediction)[i] >= 0.5:
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
            the thread running the model's train() method.
        Raises:
            TypeError: if train_thread is not of type threading.Thread.

        """
        if not train_thread.is_alive():

            # Plot losses of model training
            graph: Figure.axes = self.loss_figure.add_subplot(111)
            graph.set_title(f"Learning rate: " +
                            f"{self.deep_model.learning_rate}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.deep_model.train_losses))
            self.loss_canvas.get_tk_widget().pack(side="left")
            
            # Start predicting thread
            self.model_status_label.configure(
                            fg='green',
                            text="Using trained weights and biases to predict"
                            )
            predict_thread: threading.Thread = threading.Thread(
                                             target=self.deep_model.predict
                                             )
            predict_thread.start()
            self.manage_predicting(predict_thread=predict_thread)
        else:
            self.after(1_000, self.manage_training, train_thread)

    def start_training(self) -> None:
        """Start training model in new thread."""
        self.train_button['state'] = 'disabled'

        # Validate hidden layers shape input
        hidden_layers_shape_input = self.hidden_layers_shape_entry.get().replace(' ', '').split(',')
        for layer in hidden_layers_shape_input:
            if not layer.isdigit():
                self.model_status_label.configure(
                                        text="Invalid hidden layers shape",
                                        fg='red'
                                        )
                self.train_button['state'] = 'normal'
                return
        
        # Reset canvases and figures
        self.loss_figure = Figure()
        self.loss_canvas.get_tk_widget().destroy()
        self.loss_canvas = FigureCanvasTkAgg(figure=self.loss_figure,
                                             master=self.results_frame)
        self.results_label.pack_forget()
        
        # Start training thread
        self.deep_model.learning_rate = self.learning_rate_scale.get()
        self.deep_model.hidden_layers_shape = [int(neuron_count) for neuron_count in hidden_layers_shape_input]

        self.deep_model.init_model_values()
        self.model_status_label.configure(
                                        text="Training weights and biases...",
                                        fg='red'
                                        )
        train_thread: threading.Thread = threading.Thread(
                                              target=self.deep_model.train,
                                              args=(50_000,)
                                              )
        train_thread.start()
        self.manage_training(train_thread=train_thread)