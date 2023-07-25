import math

class XorModel():
    def sigmoid(self, z):
        return 1 / (1 + math.expr(-z))

    def train(self):
        "Train weights and bias'"
        # for i in range(100):
        #     pass

    def model(self):
        bias: float = 0
        weights: list[int] = []
        inputs: list[int] = [0,1]
        self.train()

    def main(self):
        print("XOR model")
        self.model()