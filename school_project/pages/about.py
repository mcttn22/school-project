import os

import tkinter as tk
import tkinter.font as tkf

class AboutFrame(tk.Frame):
    """Frame for about page."""
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        """Initialise about frame widgets.
        
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
        
        # Setup widgets
        self.title_label: tk.Label = tk.Label(master=self, bg='white',
                                             font=('Arial', 20), text="About")
        self.about_label: tk.Label = tk.Label(
           master=self,
           bg='white',
           font=('Arial', 14),
           text="Year 13 Computer Science Programming Project, " +
           "on developing Image Recognition from scratch " +
           "with Artificial Neural Networks,\n" +
           "then applying the models to problems, " +
           "such as recognising letter images.\n" +
           "- Max Cotton"
                                  )
        self.theory_button: tk.Button = tk.Button(master=self,
                                                  width=13,
                                                  height=1,
                                                  font=tkf.Font(size=12),
                                                  text="View Theory")
        if os.name == 'posix':
            self.theory_button.configure(command=lambda: os.system(
                                                    r'open docs/models/ann.pdf'
                                                    ))
        elif os.name == 'nt':
            self.theory_button.configure(command=lambda: os.system(
                                                         r'.\docs\models\ann.pdf'
                                                         ))
        self.theory_label: tk.Label = tk.Label(master=self,
                                               bg='white',
                                               font=('Arial', 14),
                                               text=self.load_theory(),
                                               justify='left')
        
        # Pack widgets
        self.title_label.pack()
        self.about_label.pack(pady=(10,0))
        self.theory_button.pack(pady=(10,0))
        self.theory_label.pack(pady=(10,0))
        
        # Setup frame attributes
        self.pack_propagate(False)

    def load_theory(self) -> str:
        """Load contents of 'docs/about.txt' file.
        
        Returns:
            contents of 'docs/about.txt' file as a single string.
        Raises:
            FileNotFoundError: if file does not exist.

        """
        with open(file=r'docs/about.txt', mode='r') as file:
            return ''.join(file.readlines())