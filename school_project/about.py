import tkinter as tk

class AboutFrame(tk.Frame):
    "Frame for about page"
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        "Inititalise about page widgets"
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Widgets
        self.titleLabel: tk.Label = tk.Label(self, bg="white", font=("Arial", 20), text="About")
        self.aboutLabel: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text="Year 13 Computer Science Programming Project on learning Artificial Neural Networks and their applications\n- Max Cotton")
        self.theoryLabel: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text=self.load_theory(), justify="left")
        # Pack widgets
        self.titleLabel.pack()
        self.aboutLabel.pack()
        self.theoryLabel.pack()
        # Setup
        self.pack_propagate(False)

    def load_theory(self) -> str:
        "Return contents of about.txt file"
        with open("docs/about.txt","r") as file:
            return "".join(file.readlines())