import math

class XorModel():
    "ANN model that trains to predict the output of a XOR gate with two inputs"
    def __init__(self) -> None:
        "Initialise model values"
        self.biases: list[float, float] = [0,0]
        self.weights: list[float, float] = [0,0]
        self.inputs: list[int] = [0,1]

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Biases: {', '.join(str(i) for i in self.biases)}\nWeights: {', '.join(str(i) for i in self.weights)}"

    def predict(self):
        "Use trained weights and biases to predict ouput of XOR gate on two inputs"
        print("\n*** Using trained weights and biases to predict output of XOR gate on two inputs ***\n")
        print("\nFinal state of model:\n")
        print(self)

    def sigmoid(self, z: int) -> int:
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + math.expr(-z))

    def train(self):
        "Train weights and biases"
        print("\n*** Training Weights and Biases ***\n")
        # for i in range(100):
        #     pass

    def model(self):
        "Manages model"
        self.train()
        self.predict()

    def main(self):
        "Entrypoint of XOR model"
        print("XOR model")
        self.model()