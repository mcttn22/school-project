"""Tkinter frames for loading a saved Artificial Neural Network Model."""

import sqlite3
import tkinter as tk
import tkinter.font as tkf

class LoadModelFrame(tk.Frame):
    """Frame for load model page."""
    def __init__(self, root: tk.Tk,
                 width: int, height: int,
                 bg: str, connection: sqlite3.Connection,
                 cursor: sqlite3.Cursor, dataset: str) -> None:
        """Initialise load model frame widgets.

        Args:
            root (tk.Tk): the widget object that contains this widget.
            width (int): the pixel width of the frame.
            height (int): the pixel height of the frame.
            bg (str): the hex value or name of the frame's background colour.
            connection (sqlite3.Connection): the database connection object.
            cursor (sqlite3.Cursor): the database cursor object.
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

        # Setup load model frame variables
        self.connection = connection
        self.cursor = cursor
        self.dataset = dataset
        self.use_gpu: bool
        self.model_options = self.load_model_options()

        # Setup widgets
        self.title_label = tk.Label(master=self,
                                    bg=self.BG,
                                    font=('Arial', 20),
                                    text=dataset)
        self.about_label = tk.Label(
                    master=self,
                    bg=self.BG,
                    font=('Arial', 14),
                    text=f"Load a pretrained model for the {dataset} dataset."
                    )
        self.model_status_label = tk.Label(master=self,
                                           bg=self.BG,
                                           font=('Arial', 15))

        # Don't give loaded model options if no models have been saved for the
        # dataset.
        if len(self.model_options) > 0:
            self.model_option_menu_label = tk.Label(
                                                master=self,
                                                bg=self.BG,
                                                font=('Arial', 14),
                                                text="Select a model to load or delete:"
                                                )
            self.model_option_menu_var = tk.StringVar(
                                                   master=self,
                                                   value=self.model_options[0]
                                                   )
            self.model_option_menu = tk.OptionMenu(
                                                    self,
                                                    self.model_option_menu_var,
                                                    *self.model_options
                                                    )
            self.use_gpu_check_button_var = tk.BooleanVar()
            self.use_gpu_check_button = tk.Checkbutton(
                                        master=self,
                                        width=7, height=1,
                                        font=tkf.Font(size=12),
                                        text="Use GPU",
                                        variable=self.use_gpu_check_button_var
                                        )
        else:
            self.model_status_label.configure(
                                     text='No saved models for this dataset.',
                                     fg='red'
                                     )

        # Pack widgets
        self.title_label.grid(row=0, column=0, columnspan=3)
        self.about_label.grid(row=1, column=0, columnspan=3)
        if len(self.model_options) > 0:  # Check if options should be given
            self.model_option_menu_label.grid(row=2, column=0, padx=(0,30), pady=(30,0))
            self.use_gpu_check_button.grid(row=2, column=2, rowspan=2, pady=(30,0))
            self.model_option_menu.grid(row=3, column=0, padx=(0,30), pady=(10,0))
        self.model_status_label.grid(row=4, column=0,
                                     columnspan=3, pady=50)

    def load_model_options(self) -> list[str]:
        """Load the model options from the database.

           Returns:
                a list of the model options.
        """
        sql = f"""
        SELECT Name FROM Models WHERE Dataset=?
        """
        parameters = (self.dataset.replace(" ", "_"),)
        self.cursor.execute(sql, parameters)

        # Save the string value contained within the tuple of each row
        model_options = []
        for model_option in self.cursor.fetchall():
            model_options.append(model_option[0])

        return model_options

    def load_model(self) -> object:
        """Create model using saved weights and biases.

           Returns:
               a Model object.

        """
        self.use_gpu = self.use_gpu_check_button_var.get()

        # Query data of selected saved model from database
        sql = """
        SELECT * FROM Models WHERE Dataset=? AND Name=?
        """
        parameters = (self.dataset.replace(" ", "_"), self.model_option_menu_var.get())
        self.cursor.execute(sql, parameters)
        data = self.cursor.fetchone()
        hidden_layers_shape_input = [layer for layer in data[3].replace(' ', '').split(',') if layer != '']

        # Create Model
        if not self.use_gpu:
            if self.dataset == "MNIST":
                from school_project.models.cpu.mnist import MNISTModel as Model
            elif self.dataset == "Cat Recognition":
                from school_project.models.cpu.cat_recognition import CatRecognitionModel as Model
            elif self.dataset == "XOR":
                from school_project.models.cpu.xor import XORModel as Model
            model = Model(
                hidden_layers_shape=[int(neuron_count) for neuron_count in hidden_layers_shape_input],
                train_dataset_size=data[6],
                learning_rate=data[4],
                use_relu=data[7]
                )
            model.load_model_values(file_location=data[2])

        else:
            try:
                if self.dataset == "MNIST":
                    from school_project.models.gpu.mnist import MNISTModel as Model
                elif self.dataset == "Cat Recognition":
                    from school_project.models.gpu.cat_recognition import CatRecognitionModel as Model
                elif self.dataset == "XOR":
                    from school_project.models.gpu.xor import XORModel as Model
                model = Model(
                    hidden_layers_shape=[int(neuron_count) for neuron_count in hidden_layers_shape_input],
                    train_dataset_size=data[6],
                    learning_rate=data[4],
                    use_relu=data[7]
                    )
                model.load_model_values(file_location=data[2])
            except ImportError as ie:
                self.model_status_label.configure(
                                              text="Failed to initialise GPU",
                                              fg='red'
                                              )
                raise ImportError
        return model
