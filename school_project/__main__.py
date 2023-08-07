import tkinter as tk
import tkinter.font as tkf
from experiments import Experiments
from home_page import HomePage
from image_recognition import ImageRecognition

class SchoolProject(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root.title("School Project")
        # School project variables
        self.currentPage: int = 0
        self.pages: list[tk.Frame] = [HomePage(root=self, width=self.WIDTH - 100, height=self.HEIGHT),
                                    ImageRecognition(root=self, width=self.WIDTH - 100, height=self.HEIGHT),
                                    Experiments(root=self, width=self.WIDTH - 100, height=self.HEIGHT)]
        # Widgets
        self.menu = tk.Frame(self, height=self.HEIGHT, width=self.WIDTH, bg="white")
        self.menuButtons: list[tk.Button] = [tk.Button(self.menu, width=13, height=1, text="Home", command=lambda: self.load_page(index=0), font=tkf.Font(size=12)),
                                            tk.Button(self.menu, width=13, height=1, text="Image Recognition", command=lambda: self.load_page(index=1), font=tkf.Font(size=12)),
                                            tk.Button(self.menu, width=13, height=1, text="Experiments", command=lambda: self.load_page(index=2), font=tkf.Font(size=12))]
        # Pack Widgets
        ## Pack menu
        self.menu.pack(side="left", fill="y")
        for button in self.menuButtons:
            button.pack(fill="y", expand=True)
        ## Pack homepage
        self.pages[self.currentPage].pack(side="right", fill="both", expand=True)
        # Setup
        self.pack_propagate(False)

    def load_page(self, index: int):
        "Unpack current page, then pack new page"
        # Unpack current frame
        self.pages[self.currentPage].pack_forget()
        # Pack new frame
        self.currentPage = index
        self.pages[self.currentPage].pack(side="right", fill="both", expand=True)

def main() -> None:
    "Entrypoint of project"
    root = tk.Tk()
    schoolProject = SchoolProject(root, 800, 800)
    schoolProject.pack(side="top", fill="both", expand=True)
    root.mainloop()
    # print("XOR model")
    # xorModel = XorModel()
    # xorModel.train(epochs=50_000)
    # xorModel.predict()
    # print("Image model")
    # imageModel = ImageModel()
    # imageModel.train(epochs=5_000)
    # imageModel.predict()

if __name__ == "__main__":
    main()