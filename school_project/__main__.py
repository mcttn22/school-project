import tkinter as tk
import tkinter.font as tkf
from experiments import ExperimentsFrame
from about import AboutFrame
from image_recognition import ImageRecognitionFrame

class SchoolProject(tk.Frame):
    "Main frame of school project"
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        "Initialise school project pages"
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root.title("School Project")
        # School project variables
        self.currentPage: int = 0
        self.pages: list[tk.Frame] = [AboutFrame(root=self, width=self.WIDTH - 100, height=self.HEIGHT),
                                      ImageRecognitionFrame(root=self, width=self.WIDTH - 100, height=self.HEIGHT),
                                      ExperimentsFrame(root=self, width=self.WIDTH - 100, height=self.HEIGHT)]
        # Widgets
        self.menuFrame: tk.Frame = tk.Frame(self, height=self.HEIGHT, width=self.WIDTH, bg="white")
        self.menuButtons: list[tk.Button] = [tk.Button(self.menuFrame, width=13, height=1, text="About", command=lambda: self.load_page(index=0), font=tkf.Font(size=12)),
                                             tk.Button(self.menuFrame, width=13, height=1, text="Image Recognition", command=lambda: self.load_page(index=1), font=tkf.Font(size=12)),
                                             tk.Button(self.menuFrame, width=13, height=1, text="Experiments", command=lambda: self.load_page(index=2), font=tkf.Font(size=12))]
        # Pack Widgets
        ## Pack menu
        self.menuFrame.pack(side="left", fill="y")
        for button in self.menuButtons:
            button.pack(fill="y", expand=True)
        ## Pack homepage
        self.pages[self.currentPage].pack(side="right", fill="both", expand=True, pady=(50,0))
        # Setup
        self.pack_propagate(False)

    def load_page(self, index: int) -> None:
        "If new page is different to current, unpack current page, then pack new page"
        if index != self.currentPage:
            # Unpack current frame
            self.pages[self.currentPage].pack_forget()
            # Pack new frame
            self.currentPage = index
            self.pages[self.currentPage].pack(side="right", fill="both", expand=True, pady=(50,0))

def main() -> None:
    "Entrypoint of project"
    root = tk.Tk()
    schoolProject = SchoolProject(root=root, width=1400, height=800)
    schoolProject.pack(side="top", fill="both", expand=True)
    root.mainloop()
    # Stop models training when GUI closes
    schoolProject.pages[1].catModel.running = False
    schoolProject.pages[2].xorModel.running = False

if __name__ == "__main__":
    main()