import tkinter as tk
import tkinter.font as tkf

from pages.about import AboutFrame
from pages.experiments import ExperimentsFrame
from pages.cat_recognition import CatRecognitionFrame

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
        
        # Setup school project variables
        self.current_page: int = 0
        self.pages: list[tk.Frame] = [AboutFrame(root=self,
                                                 width=self.WIDTH - 100,
                                                 height=self.HEIGHT),
                                      CatRecognitionFrame(
                                                       root=self,
                                                       width=self.WIDTH - 100,
                                                       height=self.HEIGHT
                                                       ),
                                      ExperimentsFrame(root=self,
                                                       width=self.WIDTH - 100,
                                                       height=self.HEIGHT)]
        
        # Setup widgets
        self.menu_frame: tk.Frame = tk.Frame(master=self,
                                             height=self.HEIGHT,
                                             width=self.WIDTH,
                                             bg='white')
        self.menu_buttons: list[tk.Button] = [
                                    tk.Button(
                                      master=self.menu_frame,
                                      width=12,
                                      height=1,
                                      text="About",
                                      command=lambda: self.load_page(index=0),
                                      font=tkf.Font(size=12)),
                                    tk.Button(
                                      master=self.menu_frame,
                                      width=12,
                                      height=1,
                                      text="Cat Recognition",
                                      command=lambda: self.load_page(index=1),
                                      font=tkf.Font(size=12)
                                      ),
                                    tk.Button(
                                      master=self.menu_frame,
                                      width=12,
                                      height=1,
                                      text="Experiments",
                                      command=lambda: self.load_page(index=2),
                                      font=tkf.Font(size=12)
                                      )
                                    ]
        
        # Pack menu widget
        self.menu_frame.pack(side='left', fill='y')
        for button in self.menu_buttons:
            button.pack(fill='y', expand=True)
        
        # Pack homepage widget
        self.pages[self.current_page].pack(side='right', fill='both',
                                           expand=True, pady=(50,0))
        
        # Setup frame attributes
        self.pack_propagate(False)

    def load_page(self, index: int) -> None:
        """Unpack current page and pack new page,
           if the new page is different to the current page.
        
        Args:
            index (int):
            the index of the new page to pack from the pages array.
        Raises:
            TypeError: if index is not an integer.
            IndexError: if index is not in range of the pages array.

        """
        if index != self.current_page:
            
            # Unpack current frame
            self.pages[self.current_page].pack_forget()
            
            # Pack new frame
            self.current_page = index
            self.pages[self.current_page].pack(side='right', fill='both',
                                               expand=True, pady=(50,0))

def main() -> None:
    """Entrypoint of project."""
    root = tk.Tk()
    school_project = SchoolProjectFrame(root=root, width=1430, height=890)
    school_project.pack(side='top', fill='both', expand=True)
    root.mainloop()
    
    # Stop models training when GUI closes
    school_project.pages[1].perceptron_model.running = False
    school_project.pages[2].shallow_model.running = False

if __name__ == "__main__":
    main()