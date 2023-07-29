import h5py
import matplotlib.pyplot as plt
import numpy as np

class ImageModel():
    "ANN model that trains to predict if an image is a cat or not a cat"
    def __init__(self) -> None:
        "Initialise model values"
        # Load datasets
        self.trainInputs, self.trainOutputs, self.testInputs = self.load_datasets()
        # Initialise weights and bias to 0
        self.weights = np.zeros(shape=(self.trainInputs.shape[0], 1))
        self.bias: float = 0
        self.LEARNING_RATE: float = 0.5

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Weights: {self.weights.tolist()}\nLearning Rate: {self.LEARNING_RATE}"
    
    def load_datasets(self): # TODO Not all dataset variables are used
        # Load datasets, h5 file stores large amount of data with quick access
        trainDataset = h5py.File('school_project/datasets/train-cat.h5', "r")
        testDataset = h5py.File('school_project/datasets/test-cat.h5', "r")

        # Input arrays, containing the RGB values for each pixel in each 64x64 pixel image, for 209 images
        trainInputs = np.array(trainDataset["train_set_x"][:])
        testInputs = np.array(testDataset["test_set_x"][:])

        # Output arrays, 1 for cat, 0 for not cat
        trainOutputs = np.array(trainDataset["train_set_y"][:])
        testOutputs = np.array(testDataset["test_set_y"][:])

        # Reshape input arrays into 1 dimension (flatten), then divide by 255 (RGB) to standardize them TODO
        trainInputs = trainInputs.reshape((trainInputs.shape[0], -1)).T / 255
        testInputs = testInputs.reshape((testInputs.shape[0], -1)).T / 255

        # Reshape output arrays TODO
        trainOutputs = trainOutputs.reshape((1, trainOutputs.shape[0]))
        testOutputs = testOutputs.reshape((1, testOutputs.shape[0]))

        # Load names for test dataset images
        testNames = np.array(testDataset["list_classes"][:])

        return trainInputs, trainOutputs, testInputs

    def sigmoid(self, z):
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, prediction) -> None:
        "Adjust the weights and bias via gradient descent"
        weightGradient = np.dot(self.trainInputs, (prediction - self.trainOutputs).T) / self.trainInputs.shape[1]   # TODO: Why divide by number of inputs 
        biasGradient = np.sum(prediction - self.trainOutputs) / self.trainInputs.shape[1]
        # Update weights and bias
        self.weights -= self.LEARNING_RATE * weightGradient
        self.bias -= self.LEARNING_RATE * biasGradient

    def forward_propagation(self):
        "Generate a prediction with the weights and bias, returns a prediction"
        z1 = np.dot(self.weights.T, self.trainInputs) + self.bias
        prediction = self.sigmoid(z1)
        return prediction

    def predict(self) -> None:
        "Use trained weights and bias to predict if image is a cat or not a cat"
        print("\n*** Using trained weights to predict output of XOR gate on two inputs ***\n")
        print("\nFinal state of model:")
        print(self)

    def train(self, epochs: int) -> None:
        "Train weights"
        print("\n*** Training Weights ***\n")
        losses: list[float] = []
        for epoch in range(epochs):
            prediction = self.forward_propagation()
            loss = - (1/self.trainInputs.shape[1]) * np.sum(self.trainOutputs * np.log(prediction) + (1 - self.trainOutputs) * np.log(1 - prediction))
            losses.append(loss)
            self.back_propagation(prediction=prediction)
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.show()