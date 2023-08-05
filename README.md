# Year 13 Computer Science Programming Project
## About
This project focuses on learning Artificial Neural Networks and their applications

## Run
Run with 'python3 school_project' and option '--xor' to run the XOR model or '--image' to run the Image model

## Dev Setup
- Create a virtualenv with 'python3 -m venv {venv name}'
- Use 'source ./venv/bin/activate' to enter venv
- Use pip as normal in the venv
- Use 'python3 setup.py develop' to setup preferences of setup.py (install package to venv in developing mode)
### Tests
- Use 'python3 -m unittest discover school_project/test/' to run tests
### Documentation
- XOR model: https://medium.com/analytics-vidhya/coding-a-neural-network-for-xor-logic-classifier-from-scratch-b90543648e8a
- Image model: https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e

## TODO
- Add UI (tkinter (click on image to test)), focus on image recognition, include XOR model (comparing no layer vs layer (accuracy)) and layers in description of how ANNs work