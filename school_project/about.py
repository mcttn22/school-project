import tkinter as tk
import tkinter.font as tkf

class About(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Home page variables
        # Widgets
        self.theory: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text=self.load_theory(), justify="left")
        # Pack widgets
        self.theory.pack()
        # Setup
        self.pack_propagate(False)

    def load_theory(self):
        with open("docs/README.md","r") as file:
            return "".join(file.readlines())