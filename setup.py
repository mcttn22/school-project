from setuptools import setup, find_packages

setup(
    name='school-project',
    version='2.0.0',
    packages=find_packages(),
    url='https://github.com/mcttn22/school-project.git',
    author='Max Cotton',
    author_email='maxcotton22@gmail.com',
    description='Year 13 Computer Science Programming Project',
    install_requires=[
                      'cupy-cuda12x',
                      'h5py',
                      'matplotlib',
                      'numpy==1.26.4'
    ],
)
