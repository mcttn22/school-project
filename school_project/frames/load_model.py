import sqlite3
import tkinter as tk

class LoadModelFrame(tk.Frame):
    """Frame for load model page."""
    def __init__(self, root: tk.Tk, width: int, 
                 height: int, dataset: str) -> None:
        """Initialise load model frame widgets.
        
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
        
        # Setup load model frame variables
        self.dataset = dataset
        self.use_gpu: bool
        self.model_options = self.load_model_options(dataset=dataset)
        
        # Setup widgets
        self.title_label: tk.Label = tk.Label(master=self,
                                              bg='white',
                                              font=('Arial', 20),
                                              text=dataset)
        self.about_label: tk.Label = tk.Label(
                             master=self,
                             bg='white',
                             font=('Arial', 14),
                             text=f"Load a pretrained model for the {dataset} dataset."
                             )
        self.model_option_menu_var: tk.StringVar = tk.StringVar(
                                                       master=self
                                                       )
        self.model_option_menu: tk.OptionMenu = tk.OptionMenu(
                                                 self,
                                                 self.model_option_menu_var,
                                                 *self.load_model_options(dataset=dataset)
                                                 )
        self.model_status_label: tk.Label = tk.Label(master=self,
                                                     bg='white',
                                                     font=('Arial', 15))
        
        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=3)
        self.about_label.grid(row=1, column=0, columnspan=3)
        self.model_option_menu.grid(row=2, column=1)
        self.model_status_label.grid(row=3, column=0,
                                     columnspan=3, pady=50)
        
    def load_model_options(self, dataset: str) -> list[str]:
        """Load the model options from the database.
           
           Args:
               dataset (str): the name of the dataset to load load models
               for. ('MNIST', 'Cat Recognition' or 'XOR')
            Returns:
                a list of the model options.
        """
        raise NotImplementedError
        
    def load_model(self):
        """Create model using saved weights and biases."""
        raise NotImplementedError