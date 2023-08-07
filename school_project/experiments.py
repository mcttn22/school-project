import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

class XorModel():
    "ANN model that trains to predict the output of a XOR gate with two inputs"
    def __init__(self, numInputs=2, numHiddenNeurons=4, numOutputNeurons=1) -> None:
        "Initialise model values"
        self.numHiddenNeurons = numHiddenNeurons
        # Setup pseudo random values for weight arrays
        np.random.seed(2)
        self.hiddenWeights = np.random.rand(numHiddenNeurons, numInputs)
        self.outputWeights = np.random.rand(numOutputNeurons, numHiddenNeurons)
        self.LEARNING_RATE: float = 0.1
        self.inputs = np.array([[0,0,1,1],
                               [0,1,0,1]])
        self.outputs = np.array([[0,1,1,0]])

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Number of hidden neurons: {self.numHiddenNeurons}\nHidden Weights: {self.hiddenWeights.tolist()}\nOutput Weights: {self.outputWeights.tolist()}\nLearning Rate: {self.LEARNING_RATE}"

    def sigmoid(self, z):
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, hiddenOutput, prediction) -> None:
        "Adjust the weights via gradient descent"
        outputWeightGradient = np.dot(prediction - self.outputs, hiddenOutput.T) / self.inputs.shape[1]
        hiddenWeightGradient = np.dot(np.dot(self.outputWeights.T, prediction - self.outputs) * hiddenOutput * (1 - hiddenOutput), self.inputs.T) / self.inputs.shape[1]
        # Reshape arrays to match the weight arrays for multiplication
        outputWeightGradient = np.reshape(outputWeightGradient, self.outputWeights.shape)
        hiddenWeightGradient = np.reshape(hiddenWeightGradient, self.hiddenWeights.shape)
        # Update weights
        self.hiddenWeights -= self.LEARNING_RATE * hiddenWeightGradient
        self.outputWeights -= self.LEARNING_RATE * outputWeightGradient

    def forward_propagation(self):
        "Generate a prediction with the weights, returns the hidden layer output and a prediction"
        z1 = np.dot(self.hiddenWeights, self.inputs)
        hiddenOutput = self.sigmoid(z1)
        z2 = np.dot(self.outputWeights, hiddenOutput)
        prediction = self.sigmoid(z2)
        return hiddenOutput, prediction

    def predict(self) -> None:
        "Use trained weights to predict ouput of XOR gate on two inputs"
        print("\n*** Using trained weights to predict output of XOR gate on two inputs ***\n")
        hiddenOutput, prediction = self.forward_propagation()
        # Calculate performance of model
        accuracy = 100 - np.mean(np.abs(prediction - self.outputs)) * 100
        print(f"Prediction accuracy: {round(accuracy)}%\n")
        # Output results
        print("Example results:")
        for i in range(self.inputs.shape[1]):
            print(f"{self.inputs[0][i]},{self.inputs[1][i]} = {1 if np.squeeze(prediction)[i] >= 0.5 else 0}")
        print("\nFinal state of model:")
        print(self)

    def train(self, epochs: int) -> None:
        "Train weights"
        print("\n*** Training Weights ***\n")
        losses: list[float] = []
        for epoch in range(epochs):
            hiddenOutput, prediction = self.forward_propagation()
            loss = - (1/self.inputs.shape[1]) * np.sum(self.outputs * np.log(prediction) + (1 - self.outputs) * np.log(1 - prediction))
            losses.append(loss)
            self.back_propagation(hiddenOutput=hiddenOutput, prediction=prediction)
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.show()

class Experiments(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="blue")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Experiments variables
        # Widgets
        # Setup
        self.pack_propagate(False)