### Todo

- Code
  - Is dataset / does it need to be normalised, or need NOT NULL specification?
  - Make UI nicer? (Use customtkinter?)
  - Memory Leak issue
  - Maybe
    - Save optimum models
    - Give option to continue training pretrained model
    - Add ability to add external .npz file

- Report
  - Finish design draft
    - See emailed NEA guide on algorithms (ANN, linked list, recursive)
    - Talk about github
      - Features (automated testing (then show actual tests in test section), readme (mention here then show implementation in technical solution?), installing project and dependencies, gitignore?, license?)
      - commits + branches
    - Add file structure of all files
    - Maybe add attributes and methods to class diagrams
    - HCI
  - Technical solution
    - Maybe use Jupyter notebook
    - Tkinter manage method is an example of recursion?
  - Testing
    - Display train-dataset-size - accuracy graph for testing? etc
  - Report should be objective focused (technical solution and testing)
  - Mention time complexity in report, speed times of saving and loading, and why cat accuracy is low

- Update setup.py version

- Notes
  - Cat dataset has 50 test images (fewer needed for testing) and MNIST has 10,000

- Does comparing hyper-parameter graphs + accuracy go in test section? Dr.Grey says test section

- Objective trace table (obj, brief description, link to section in document, maybe link to where tested)
- Remove memory leak mentioning
Dr.Grey Qs:
  - Not all 'show not tell', try to give comments/explanations in report that make it obvious (recursive methods)
  - Docstrings and comments enough? yes but need to say this section does this objective
  - Should test frames, test in file structure, test methods in school_project class and test instructions etc be in technical solution or testing section? yes~
  - compare gpu vs cpu time?