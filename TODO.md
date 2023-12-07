### Todo

- Code
  - More tests
    - assert shape of network (correct number of layers and neurons in each layer (size of output is correct for next layer's input))
      - assert shape matches shape set with ui
      - assert n rows of first matrix = n columns of second matrix or whatever
    - assert each layer's transfer function
    - assert learning rate of each layer is the same
    - assert train dataset size
    - assert derivative weights/biases = weights/biases shape (+ remove assert in implementation)
    - test derivative functions
  - Make UI nicer? (Use customtkinter?)
  - Memory Leak issue
  - Maybe
    - Save optimum models
    - Give option to continue training pretrained model
    - Add ability to add external .npz file

- Report
  - Finish design draft
    - Maybe add attributes and methods to class diagrams
    - HCI
  - Technical solution
    - Maybe use Jupyter notebook
    - Tkinter manage method is an example of recursion?
  - Testing
    - Display train-dataset-size - accuracy graph for testing? etc
  - Mention time complexity in report, speed times of saving and loading, and why cat accuracy is low

- Update setup.py version

- Notes
  - Cat dataset has 50 test images (fewer needed for testing) and MNIST has 10,000