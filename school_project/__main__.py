import threading
import uuid

import sqlite3
import tkinter as tk
import tkinter.font as tkf

from school_project.frames.create_model import (HyperParameterFrame,
                                                TrainingFrame)
from school_project.frames.load_model import LoadModelFrame
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
        self.load_model_frame: LoadModelFrame
        self.test_frame: TestMNISTFrame | TestCatRecognitionFrame | TestXORFrame
        self.connection, self.cursor = self.setup_database()
        self.model = None

        # Record if the model should be saved after testing,
        # as only newly created models should be given the option to be saved.
        self.saving_model: bool

        # Setup school project frame widgets
        self.exit_hyper_parameter_frame_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Exit",
              command=self.exit_hyper_parameter_frame
              )
        self.exit_load_model_frame_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Exit",
              command=self.exit_load_model_frame
              )
        self.train_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Train Model",
              command=self.enter_training_frame
              )
        self.stop_training_button: tk.Button = tk.Button(
              master=self,
              width=15, height=1,
              font=tkf.Font(size=12),
              text="Stop Training Model",
              command=self.stop_training
              )
        self.test_created_model_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Test Model",
              command=self.test_created_model
              )
        self.test_loaded_model_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Test Model",
              command=self.test_loaded_model
              )
        self.save_model_label: tk.Label = tk.Label(
                                  master=self,
                                  text="Enter a name for your trained model:",
                                  bg='white',
                                  font=('Arial', 15)
                                  )
        self.save_model_name_entry: tk.Entry = tk.Entry(master=self,
                                                        width=13)
        self.save_model_button: tk.Button = tk.Button(
              master=self,
              width=13,
              height=1,
              font=tkf.Font(size=12),
              text="Save Model",
              command=self.save_model
              )
        self.exit_button: tk.Button = tk.Button(
              master=self,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Exit",
              command=self.enter_home_frame
              )

        # Setup home frame
        self.home_frame: tk.Frame = tk.Frame(master=self, 
                                            width=self.WIDTH, 
                                            height=self.HEIGHT,
                                            bg='white')
        self.title_label: tk.Frame = tk.Label(
                       master=self.home_frame, bg='white',
                       font=('Arial', 20), 
                       text="A-level Computer Science NEA Programming Project"
                       )
        self.about_label: tk.Label = tk.Label(
           master=self.home_frame,
           bg='white',
           font=('Arial', 14),
           text="An investigation into how Artificial Neural Networks work, " +
           "the effects of their hyper-parameters and their applications " +
           "in Image Recognition.\n\n" +
           " - Max Cotton"
           )
        self.model_menu_label: tk.Label = tk.Label(
                                          master=self.home_frame,
                                          bg='white',
                                          font=('Arial', 14),
                                          text="Create a new model " +
                                          "or load a pre-trained model "
                                          "for one of the following datasets:"
                                          )
        self.dataset_option_menu_var: tk.StringVar = tk.StringVar(
                                                       master=self.home_frame,
                                                       value="MNIST"
                                                       )
        self.dataset_option_menu: tk.OptionMenu = tk.OptionMenu(
                                                 self.home_frame,
                                                 self.dataset_option_menu_var,
                                                 "MNIST",
                                                 "Cat Recognition",
                                                 "XOR"
                                                 )
        self.create_model_button: tk.Button = tk.Button(
              master=self.home_frame,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Create Model",
              command=self.enter_hyper_parameter_frame
              )
        self.load_model_button: tk.Button = tk.Button(
              master=self.home_frame,
              width=13, height=1,
              font=tkf.Font(size=12),
              text="Load Model",
              command=self.enter_load_model_frame
              )
        
        # Grid home frame widgets
        self.title_label.grid(row=0, column=0, columnspan=4, pady=(10,0))
        self.about_label.grid(row=1, column=0, columnspan=4, pady=(10,50))
        self.model_menu_label.grid(row=2, column=0, columnspan=4)
        self.dataset_option_menu.grid(row=3, column=0, columnspan=4, pady=30)
        self.create_model_button.grid(row=4, column=1)
        self.load_model_button.grid(row=4, column=2)

        self.home_frame.pack()
        
        # Setup frame attributes
        self.grid_propagate(flag=False)
        self.pack_propagate(flag=False)

    def setup_database(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Create a connection to the pretrained_models database file and 
           setup base table if needed.
           
           Returns:
               a tuple of the database connection and the cursor for it.
        
        """
        connection = sqlite3.connect('school_project/pretrained_models.db')
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Pretrained_Models
        (Model_Name TEXT PRIMARY KEY,
        Dataset_Name TEXT,
        File_Location TEXT,
        Use_ReLu INTEGER)
        """)
        return (connection, cursor)

    def enter_hyper_parameter_frame(self) -> None:
        """Unpack home frame and pack hyper-parameter frame."""
        self.home_frame.pack_forget()
        self.hyper_parameter_frame = HyperParameterFrame(
                    root=self,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    dataset=self.dataset_option_menu_var.get()
                    )
        self.hyper_parameter_frame.pack()
        self.train_button.pack()
        self.exit_hyper_parameter_frame_button.pack(pady=(10,0))

    def enter_load_model_frame(self) -> None:
        """Unpack home frame and pack load model frame."""
        self.home_frame.pack_forget()
        self.load_model_frame = LoadModelFrame(
                        root=self,
                        width=self.WIDTH,
                        height=self.HEIGHT,
                        dataset=self.dataset_option_menu_var.get()
                        )
        self.load_model_frame.pack()
        self.test_loaded_model_button.pack()
        self.exit_load_model_frame_button.pack(pady=(10,0))
        
    def exit_hyper_parameter_frame(self) -> None:
        """Unpack hyper-parameter frame and pack home frame."""
        self.hyper_parameter_frame.pack_forget()
        self.train_button.pack_forget()
        self.exit_hyper_parameter_frame_button.pack_forget()
        self.home_frame.pack()

    def exit_load_model_frame(self) -> None:
        """Unpack load model frame and pack home frame."""
        self.load_model_frame.pack_forget()
        self.test_loaded_model_button.pack_forget()
        self.exit_load_model_frame_button.pack_forget()
        self.home_frame.pack()

    def enter_training_frame(self) -> None:
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
        self.exit_hyper_parameter_frame_button.pack_forget()
        self.training_frame = TrainingFrame(
                                         root=self,
                                         width=self.WIDTH, 
                                         height=self.HEIGHT,
                                         model=self.model,
                                         epoch_count=self.hyper_parameter_frame.epoch_count_scale.get()
                                         )
        self.training_frame.pack()
        self.stop_training_button.pack()
        self.manage_training(train_thread=self.training_frame.train_thread)

    def stop_training(self) -> None:
        """Stop model training."""
        self.model.running = False

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
            self.training_frame.training_progress_label.pack_forget()
            self.training_frame.plot_losses(model=self.model)
            self.stop_training_button.pack_forget()
            self.test_created_model_button.pack(pady=(30,0))
        else:
            self.training_frame.training_progress_label.configure(text=self.model.training_progress)
            self.after(100, self.manage_training, train_thread)

    def test_created_model(self) -> None:
        """Unpack training frame, pack test frame for the dataset
           and begin managing the test thread."""
        self.saving_model = True
        self.training_frame.pack_forget()
        self.test_created_model_button.pack_forget()
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

    def test_loaded_model(self) -> None:
        """Unpack load model frame, pack test frame for the dataset
           and begin managing the test thread."""
        self.saving_model = False
        try:
            self.model = self.load_model_frame.load_model()
        except (ValueError, ImportError) as e:
            return
        self.load_model_frame.pack_forget()
        self.test_loaded_model_button.pack_forget()
        if self.hyper_parameter_frame.dataset == "MNIST":
            self.test_frame = TestMNISTFrame( 
                                root=self,
                                width=self.WIDTH,
                                height=self.HEIGHT,
                                use_gpu=self.load_model_frame.use_gpu,
                                model=self.model
                                )
        elif self.hyper_parameter_frame.dataset == "Cat Recognition":
            self.test_frame = TestCatRecognitionFrame(
                                root=self,
                                width=self.WIDTH, 
                                height=self.HEIGHT,
                                use_gpu=self.load_model_frame.use_gpu,
                                model=self.model
                                )
        elif self.hyper_parameter_frame.dataset == "XOR":
            self.test_frame = TestXORFrame(
                    root=self, width=self.WIDTH,
                    height=self.HEIGHT, model=self.model
                    )
        self.test_frame.pack()
        self.manage_testing(test_thread=self.test_frame.test_thread)

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
            self.test_frame.plot_results(model=self.model)
            if self.saving_model:
                self.save_model_label.pack(pady=(30,0))
                self.save_model_name_entry.pack(pady=10)
                self.save_model_button.pack()
            self.exit_button.pack(pady=(20,0))
        else:
            self.after(1_000, self.manage_testing, test_thread)

    def save_model(self) -> None:
        """Export the model, save the model information to the database, then 
           enter the home frame."""
        # Export model to random hex file name
        file_location = f"school_project/exported-models/{uuid.uuid4().hex}.npz"
        self.model.export(file_location=file_location)

        # Save the model information to the database
        params = (self.save_model_name_entry.get(), self.dataset_option_menu_var.get(), 
                  file_location, self.hyper_parameter_frame.use_relu_check_button_var.get())
        sql = """
        INSERT OR REPLACE INTO Pretrained_Models (Model_Name, Dataset_Name, File_Location, Use_ReLu)
        VALUES(?, ?, ?, ?)
        """
        self.cursor.execute(sql, params)
        self.connection.commit()

        self.enter_home_frame()

    def enter_home_frame(self) -> None:
        """Unpack test frame and pack home frame."""
        self.model = None  # Free up trained Model from memory
        self.test_frame.pack_forget()
        if self.saving_model:
            self.save_model_label.pack_forget()
            self.save_model_name_entry.delete(0, tk.END)  # Clear entry's text
            self.save_model_name_entry.pack_forget()
            self.save_model_button.pack_forget()
        self.exit_button.pack_forget()
        self.home_frame.pack()
        
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