import os
import threading

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import tkinter.font as tkf

from school_project.models.image_recognition.cat import PerceptronModel

class CatRecognitionFrame(tk.Frame):
    """Frame for Cat Recognition page."""
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
        self.perceptron_model: PerceptronModel = PerceptronModel()
        
        # Setup widgets
        self.menu_frame: tk.Frame = tk.Frame(master=self, bg='white')
        self.title_label: tk.Label = tk.Label(master=self.menu_frame,
                                              bg='white',
                                              font=('Arial', 20),
                                              text="Cat Recognition")
        self.about_label: tk.Label = tk.Label(
                                     master=self.menu_frame,
                                     bg='white',
                                     font=('Arial', 14),
                                     text="An Image model trained on " +
                                     "recognising if an image is a cat or not."
                                    )
        self.cat_recognition_theory_button: tk.Button = tk.Button(
                                          master=self.menu_frame, 
                                          width=23,
                                          height=1,
                                          font=tkf.Font(size=12),
                                          text="View Cat Recognition Theory")
        self.model_theory_button: tk.Button = tk.Button(
                                          master=self.menu_frame, 
                                          width=15,
                                          height=1,
                                          font=tkf.Font(size=12),
                                          text="View Model Theory")
        if os.name == 'posix':
            self.cat_recognition_theory_button.configure(
                                                    command=lambda: os.system(
                                 r'open docs/models/image_recognition/cat.pdf'
                                 ))
            self.model_theory_button.configure(command=lambda: os.system(
                                r'open docs/models/utils/perceptron_model.pdf'
                                ))
        elif os.name == 'nt':
            self.cat_recognition_theory_button.configure(
                                                    command=lambda: os.system(
                                      r'docs\models\image_recognition\cat.pdf'
                                      ))
            self.model_theory_button.configure(command=lambda: os.system(
                                   r'.\docs\models\utils\perceptron_model.pdf'
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
        self.learning_rate_scale.set(value=self.perceptron_model.learning_rate)
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
        self.title_label.grid(row=0, column=0, columnspan=3)
        self.about_label.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        self.cat_recognition_theory_button.grid(row=2, column=1, pady=(10,0))
        self.model_theory_button.grid(row=3, column=0, pady=(50, 0))
        self.train_button.grid(row=3, column=2, pady=(50, 0))
        self.learning_rate_scale.grid(row=4, column=0,
                                      columnspan=3, pady=(10, 0))
        self.model_status_label.grid(row=5, column=0,
                                     columnspan=3, pady=(10, 0))
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
             f"{round(self.perceptron_model.test_prediction_correctness)}%"
             )
            image1: Figure.axes = self.image_figure.add_subplot(121)
            if np.squeeze(self.perceptron_model.test_prediction)[0] >= 0.5:
                image1.set_title("Cat")
            else:
                image1.set_title("Not a cat")
            image1.imshow(
                     self.perceptron_model.test_inputs[:,0].reshape((64,64,3))
                     )
            image2: Figure.axes = self.image_figure.add_subplot(122)
            if np.squeeze(self.perceptron_model.test_prediction)[14] >= 0.5:
                image2.set_title("Cat")
            else:
                image2.set_title("Not a cat")
            image2.imshow(
                    self.perceptron_model.test_inputs[:,14].reshape((64,64,3))
                    )
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
            graph.set_title(f"Learning rate: " +
                            "{self.perceptron_model.learning_rate}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.perceptron_model.train_losses))
            self.loss_canvas.get_tk_widget().pack(side='left')
            
            # Start predicting thread
            self.model_status_label.configure(
                             text="Using trained weights and bias to predict",
                             fg='green'
                             )
            predict_thread: threading.Thread = threading.Thread(
                                          target=self.perceptron_model.predict
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
        self.perceptron_model.learning_rate = self.learning_rate_scale.get()
        self.perceptron_model.init_model_values()
        self.model_status_label.configure(text="Training weights and bias...",
                                          fg='red')
        train_thread: threading.Thread = threading.Thread(
                                           target=self.perceptron_model.train,
                                           args=(5_000,)
                                           )
        train_thread.start()
        self.manage_training(train_thread=train_thread)