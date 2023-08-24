# Year 13 Computer Science Programming Project

This project focuses on image recognition, by creating Artificial Neural Networks from scratch and applying them to problems, such as recognising numbers from images.

## Installation
Create a virtual environment (venv) with:
```
python3 -m venv {venv name}
```
Enter the venv with:
```
source ./{venv name}/bin/activate
```
Download the Repository with:
```
git clone https://github.com/mcttn22/school-project.git
```
Enter the project directory with:
```
cd school-project/
```
For normal use, install the dependencies and the project to the venv with:
```
python3 setup.py install
```

## Usage

Run with:
```
python3 school_project
```

## Development

Install the dependencies and the project to the venv in developing mode with:
```
python3 setup.py develop
```

Run Tests with:
```
python3 -m unittest discover school_project/test/
```

Compile Model Theory Documentation PDFs with:
```
make -C ./docs/models/ all
```