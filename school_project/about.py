import tkinter as tk

class About(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="red")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Home page variables
        # Widgets
        # Setup
        self.pack_propagate(False)