[![python](https://img.shields.io/badge/Python-3-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

# Year 13 Computer Science Programming Project

This project focuses on image recognition, by creating Artificial Neural Networks from scratch and applying them to problems, such as recognising numbers from images.

## Installation

1. Create a virtual environment (venv) with:
   - Windows:
     ```
     python -m venv {venv name}
     ```
   - Linux:
     ```
     python3 -m venv {venv name}
     ```

2. Enter the venv with:
   - Windows:
     ```
     .\{venv name}\Scripts\activate
     ```
   - Linux:
     ```
     source ./{venv name}/bin/activate
     ```

3. Download the Repository with:

   - ```
     git clone https://github.com/mcttn22/school-project.git
     ```
   - Or by downloading as a Zip file

</br>

4. Enter the project directory with:
   ```
   cd school-project/
   ```

5. For normal use, install the dependencies and the project to the venv with:
   - Windows:
     ```
     python setup.py install
     ```
   - Linux:
     ```
     python3 setup.py install
     ```

## Usage

Run with:
- Windows:
  ```
  python school_project
  ```
- Linux:
  ```
  python3 school_project
  ```

## Development

Install the dependencies and the project to the venv in developing mode with:
- Windows:
  ```
  python setup.py develop
  ```
- Linux:
  ```
  python3 setup.py develop
  ```

Run Tests with:
- Windows:
  ```
  python -m unittest discover .\school_project\test\
  ```
- Linux:
  ```
  python3 -m unittest discover ./school_project/test/
  ```

Compile Model Theory PDFs with:
```
make -C ./docs/models/ all
```
*Note: This requires the Latexmk library*