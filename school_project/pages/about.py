import tkinter as tk

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
           text="A-level Computer Science NEA Programming Project, " +
           "on developing Image Recognition from scratch " +
           "with Artificial Neural Networks,\n" +
           "then applying the models to problems, " +
           "such as recognising number images.\n" +
           "- Max Cotton"
                                  )
        
        # Pack widgets
        self.title_label.pack()
        self.about_label.pack(pady=(10,0))
        
        # Setup frame attributes
        self.pack_propagate(False)