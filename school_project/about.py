import tkinter as tk

class About(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Home page variables
        # Widgets
        self.title: tk.Label = tk.Label(self, bg="white", font=("Arial", 20), text="About")
        self.about: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text="Year 13 Computer Science Programming Project on learning Artificial Neural Networks and their applications\n- Max Cotton")
        self.theory: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text=self.load_theory(), justify="left")
        # Pack widgets
        self.title.pack()
        self.about.pack()
        self.theory.pack()
        self.theory.pack()
        # Setup
        self.pack_propagate(False)

    def load_theory(self):
        with open("docs/about.txt","r") as file:
            return "".join(file.readlines())