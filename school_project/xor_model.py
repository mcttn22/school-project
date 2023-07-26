import math
import numpy as np

class XorModel():
    "ANN model that trains to predict the output of a XOR gate with two inputs"
    def __init__(self) -> None:
        "Initialise model values"
        self.hidden_bias: float = 0
        self.output_bias: float = 0
        self.hidden_weight: float = 0
        self.output_weight: float = 0
        self.LEARNING_RATE: float = 0.1
        self.inputs = np.array([[0,0,1,1],
                               [0,1,0,1]])
        self.outputs = np.array([0,1,1,0])

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Biases: {', '.join(str(i) for i in self.biases)}\nWeights: {', '.join(str(i) for i in self.weights)}\nLearning Rate: {self.LEARNING_RATE}"

    def sigmoid(self, z: float) -> float:
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + math.expr(-z))

    def cost_gradient(self, cost: float) -> float:
        "Calculate the gradient of the cost function"

    def cost(self) -> float:
        "Calculate the error in the prediction"

    def back_propagation(self) -> None:
        "Adjust the weights and biases based on the cost of the prediction"
    
    def forward_propagation(self):
        "Generate a prediction with the weights and biases"

    def predict(self) -> None:
        "Use trained weights and biases to predict ouput of XOR gate on two inputs"
        print("\n*** Using trained weights and biases to predict output of XOR gate on two inputs ***\n")
        print("\nFinal state of model:\n")
        print(self)

    def train(self, epochs: int) -> None:
        "Train weights and biases"
        print("\n*** Training Weights and Biases ***\n")
        for epoch in range(epochs):
            pass