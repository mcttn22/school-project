import matplotlib.pyplot as plt
import numpy as np

class XorModel():
    "ANN model that trains to predict the output of a XOR gate with two inputs"
    def __init__(self) -> None:
        "Initialise model values"
        # Setup pseudo random values for weight arrays
        np.random.seed(2)
        self.hiddenWeights = np.random.rand(2, 2)
        self.outputWeights = np.random.rand(1, 2)
        self.LEARNING_RATE: float = 0.1
        self.inputs = np.array([[0,0,1,1],
                               [0,1,0,1]])
        self.outputs = np.array([[0,1,1,0]])

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Hidden Weights: {self.hiddenWeights}\nOutput Weights: {self.outputWeights}\nLearning Rate: {self.LEARNING_RATE}"

    def sigmoid(self, z):
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, hiddenOutput, prediction) -> None:
        "Adjust the weights and biases via gradient descent"
        outputWeightGradient = np.dot(prediction - self.outputs, hiddenOutput.T) / self.inputs.shape[1]   # Why divide by m ?
        hiddenWeightGradient = np.dot(np.dot(self.outputWeights.T, prediction - self.outputs) * hiddenOutput * (1 - hiddenOutput), self.inputs.T) / self.inputs.shape[1]
        # Reshape arrays
        outputWeightGradient = np.reshape(outputWeightGradient, self.outputWeights.shape)
        hiddenWeightGradient = np.reshape(hiddenWeightGradient, self.hiddenWeights.shape)
        # Update weights
        self.hiddenWeights -= self.LEARNING_RATE * hiddenWeightGradient
        self.outputWeights -= self.LEARNING_RATE * outputWeightGradient

    def forward_propagation(self):
        "Generate a prediction with the weights and biases, returns the hidden layer output and a prediction"
        z1 = np.dot(self.hiddenWeights, self.inputs)
        hiddenOutput = self.sigmoid(z1)
        z2 = np.dot(self.outputWeights, hiddenOutput)
        prediction = self.sigmoid(z2)
        return hiddenOutput, prediction

    def predict(self) -> None:
        "Use trained weights and biases to predict ouput of XOR gate on two inputs"
        print("\n*** Using trained weights and biases to predict output of XOR gate on two inputs ***\n")
        print("\nFinal state of model:\n")
        print(self)

    def train(self, epochs: int) -> None:
        "Train weights and biases"
        print("\n*** Training Weights and Biases ***\n")
        losses: list[float] = []
        for epoch in range(epochs):
            hiddenOutput, prediction = self.forward_propagation()
            loss = - (1/self.inputs.shape[1]) * np.sum(self.outputs * np.log(prediction) + (1 - self.outputs) * np.log(1 - prediction))
            losses.append(loss)
            self.back_propagation(hiddenOutput=hiddenOutput, prediction=prediction)
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")