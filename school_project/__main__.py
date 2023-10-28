import threading

import tkinter as tk
import tkinter.font as tkf

from school_project.frames.create_model import (HyperParameterFrame,
                                                TrainingFrame)
from school_project.frames.test_model import (TestMNISTFrame,
                                              TestCatRecognitionFrame, 
                                              TestXORFrame)

class SchoolProjectFrame(tk.Frame):
    """Main frame of school project."""
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        """Initialise school project pages.
        
        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
        Raises:
            TypeError: if root, width or height are not of the correct type.
        
        """
        super().__init__(master=root, width=width, height=height, bg='white')
        self.root = root.title("School Project")
        self.WIDTH = width
        self.HEIGHT = height
        
        # Setup school project frame variables
        self.hyper_parameter_frame: HyperParameterFrame
        self.training_frame: TrainingFrame
        self.test_frame: TestMNISTFrame | TestCatRecognitionFrame | TestXORFrame
        self.model = None

        # Setup school project frame widgets
        self.train_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Train Model",
              command=self.load_training_frame
              )
        self.test_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Test Model",
              command=self.load_test_frame
              )
        self.exit_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Exit",
              command=self.load_home_frame
              )

        # Setup home frame
        self.home_frame: tk.Frame = tk.Frame(master=self, 
                                            width=self.WIDTH, 
                                            height=self.HEIGHT,
                                            bg='white')
        self.create_model_menu_label: tk.Label = tk.Label(
                                           master=self.home_frame,
                                           bg='white',
                                           font=('Arial', 14),
                                           text="Create New Model for one " +
                                            "one of the following datasets."
                                            )
        self.create_model_dataset_option_menu_var: tk.StringVar = tk.StringVar(master=self.home_frame)
        self.create_model_dataset_option_menu: tk.OptionMenu = tk.OptionMenu(
                                    self.home_frame,
                                    self.create_model_dataset_option_menu_var,
                                    "MNIST",
                                    "Cat Recognition",
                                    "XOR"
                                    )
        self.create_model_button: tk.Button = tk.Button(
              master=self.home_frame,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Create Model",
              command=self.load_hyper_parameter_frame
              )
        self.load_model_menu_label: tk.Label = tk.Label(
                                           master=self.home_frame,
                                           bg='white',
                                           font=('Arial', 14),
                                           text="Load Pretrained Models:"
                                            )
        self.load_model_option_menu_value: tk.StringVar = tk.StringVar(
                                                       master=self.home_frame
                                                       )
        self.load_model_option_menu: tk.OptionMenu = tk.OptionMenu(
                                         self.home_frame,
                                         self.load_model_option_menu_value,
                                         *self.load_pretrained_model_options()
                                         )
        self.load_model_button: tk.Button = tk.Button(
              master=self.home_frame,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Load Model",
              command=self.test_loaded_model
              )
        
        # Grid home frame widgets
        self.create_model_menu_label.grid(row=0, column=0)
        self.create_model_dataset_option_menu.grid(row=1, column=0)
        self.create_model_button.grid(row=2, column=0)
        self.load_model_menu_label.grid(row=0, column=1, padx=(20,0))
        self.load_model_option_menu.grid(row=1, column=1, padx=(20,0))
        self.load_model_button.grid(row=2, column=1, padx=(20,0))

        self.home_frame.pack()
        
        # Setup frame attributes
        self.grid_propagate(flag=False)
        self.pack_propagate(flag=False)

    def load_hyper_parameter_frame(self) -> None:
        """Unpack home frame and pack hyper parameter frame"""
        self.home_frame.pack_forget()
        self.hyper_parameter_frame = HyperParameterFrame(
                     root=self,
                     width=self.WIDTH,
                     height=self.HEIGHT,
                     dataset=self.create_model_dataset_option_menu_var.get()
                     )
        self.hyper_parameter_frame.pack()
        self.train_button.pack()
        

    def load_pretrained_model_options(self) -> list[str]:  # TODO
        """Not Implemented"""
        return ["Not Implemented"]
    
    def test_loaded_model(self):  # TODO
        """Not Implemented"""

    def load_training_frame(self) -> None:
        """Load untrained model from hyper parameter frame,
           unpack hyper-parameter frame, pack training frame
           and begin managing the training thread.
        """
        try:
            self.model = self.hyper_parameter_frame.create_model()
        except (ValueError, ImportError) as e:
            return
        self.hyper_parameter_frame.pack_forget()
        self.train_button.pack_forget()
        self.training_frame = TrainingFrame(
                                         root=self, width=self.WIDTH, 
                                         height=self.HEIGHT, model=self.model
                                         )
        self.training_frame.pack()
        self.manage_training(train_thread=self.training_frame.train_thread)

    def manage_training(self, train_thread: threading.Thread) -> None:
        """Wait for model training thread to finish,
           then plot training losses on training frame.
        
        Args:
            train_thread (threading.Thread):
            the thread running the model's train() method.
        Raises:
            TypeError: if train_thread is not of type threading.Thread.

        """
        if not train_thread.is_alive():
            self.training_frame.plot_losses()
            self.test_button.pack()
        else:
            self.after(1_000, self.manage_training, train_thread)

    def load_test_frame(self) -> None:
        """Unpack trainig frame, pack test frame for the dataset
           and begin managing the test thread.
        """
        self.training_frame.pack_forget()
        self.test_button.pack_forget()
        if self.hyper_parameter_frame.dataset == "MNIST":
            self.test_frame = TestMNISTFrame( 
                                   root=self,
                                   width=self.WIDTH,
                                   height=self.HEIGHT,
                                   use_gpu=self.hyper_parameter_frame.use_gpu,
                                   model=self.model
                                   )
        elif self.hyper_parameter_frame.dataset == "Cat Recognition":
            self.test_frame = TestCatRecognitionFrame(
                                 root=self,
                                 width=self.WIDTH, 
                                 height=self.HEIGHT,
                                 use_gpu=self.hyper_parameter_frame.use_gpu,
                                 model=self.model
                                 )
        elif self.hyper_parameter_frame.dataset == "XOR":
            self.test_frame = TestXORFrame(
                     root=self, width=self.WIDTH,
                     height=self.HEIGHT, model=self.model
                     )
        self.test_frame.pack()
        self.manage_testing(test_thread=self.test_frame.test_thread)

    def load_home_frame(self) -> None:
        """Unpack test frame and pack home frame."""
        self.test_frame.pack_forget()
        self.exit_button.pack_forget()
        self.home_frame.pack()

    def manage_testing(self, test_thread: threading.Thread) -> None:
        """Wait for model test thread to finish,
           then plot results on test frame.
        
        Args:
            test_thread (threading.Thread):
            the thread running the model's predict() method.
        Raises:
            TypeError: if test_thread is not of type threading.Thread.
        
        """
        if not test_thread.is_alive():
            self.test_frame.plot_results()
            self.exit_button.pack()
        else:
            self.after(1_000, self.manage_testing, test_thread)
        
def main() -> None:
    """Entrypoint of project."""
    root = tk.Tk()
    school_project = SchoolProjectFrame(root=root, width=1465, height=890)
    school_project.pack(side='top', fill='both', expand=True)
    root.mainloop()
    
    # Stop model training when GUI closes
    if school_project.model != None:
        school_project.model.running = False

if __name__ == "__main__":
    main()