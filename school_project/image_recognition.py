import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import tkinter as tk
import tkinter.font as tkf

class CatModel():
    "ANN model that trains to predict if an image is a cat or not a cat"
    def __init__(self) -> None:
        "Initialise model values"
        self.trainLosses: list[float] = []
        self.prediction = None
        self.predictionAccuracy = None
        self.running: bool = True
        # Load datasets
        self.trainInputs, self.trainOutputs, self.testInputs, self.testOutputs = self.load_datasets()
        # Initialise weights and bias to 0
        self.weights = np.zeros(shape=(self.trainInputs.shape[0], 1))
        self.bias: float = 0
        self.LEARNING_RATE: float = 0.005

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Weights: {self.weights}\nBias: {self.bias}\nLearning Rate: {self.LEARNING_RATE}"
    
    def init_values(self):
        "Initialise weights, bias and training losses"
        self.weights = np.zeros(shape=(self.trainInputs.shape[0], 1))
        self.bias = 0
        self.trainLosses = []
    
    def load_datasets(self):
        # Load datasets, h5 file stores large amount of data with quick access
        trainDataset = h5py.File('school_project/datasets/train-cat.h5', "r")
        testDataset = h5py.File('school_project/datasets/test-cat.h5', "r")
        # Input arrays, containing the RGB values for each pixel in each 64x64 pixel image, for 209 images
        trainInputs = np.array(trainDataset["train_set_x"][:])
        testInputs = np.array(testDataset["test_set_x"][:])
        # Output arrays, 1 for cat, 0 for not cat
        trainOutputs = np.array(trainDataset["train_set_y"][:])
        testOutputs = np.array(testDataset["test_set_y"][:])
        # Reshape input arrays into 1 dimension (flatten), then divide by 255 (RGB) to standardize them to a number between 0 and 1
        trainInputs = trainInputs.reshape((trainInputs.shape[0], -1)).T / 255
        testInputs = testInputs.reshape((testInputs.shape[0], -1)).T / 255
        # Reshape output arrays into a 1 dimensional list of outputs
        trainOutputs = trainOutputs.reshape((1, trainOutputs.shape[0]))
        testOutputs = testOutputs.reshape((1, testOutputs.shape[0]))
        return trainInputs, trainOutputs, testInputs, testOutputs

    def sigmoid(self, z):
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, prediction) -> None:
        "Adjust the weights and bias via gradient descent"
        weightGradient = np.dot(self.trainInputs, (prediction - self.trainOutputs).T) / self.trainInputs.shape[1]
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
        # Calculate prediction for test dataset
        z1 = np.dot(self.weights.T, self.testInputs) + self.bias
        self.prediction = self.sigmoid(z1)
        # Calculate performance of model
        self.predictionAccuracy = 100 - np.mean(np.abs(self.prediction.round() - self.testOutputs)) * 100

    def train(self, epochs: int) -> None:
        "Train weights and bias"
        self.init_values()
        for epoch in range(epochs):
            if not self.running:
                break
            prediction = self.forward_propagation()
            loss = - (1/self.trainInputs.shape[1]) * np.sum(self.trainOutputs * np.log(prediction) + (1 - self.trainOutputs) * np.log(1 - prediction))
            self.trainLosses.append(np.squeeze(loss))
            self.back_propagation(prediction=prediction)

class ImageRecognition(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Image recognition variables
        self.catModel = CatModel()
        # Widgets
        self.title: tk.Label = tk.Label(self, bg="white", font=("Arial", 20), text="Image Recognition")
        self.about: tk.Label = tk.Label(self, bg="white", font=("Arial", 14), text="This is a cat model")
        self.theoryButton: tk.Button = tk.Button(self, width=13, height=1, text="View Theory", command=lambda: os.system("open docs/image_model.pdf"), font=tkf.Font(size=12))
        self.trainButton: tk.Button = tk.Button(self, width=13, height=1, text="Train Model", command=self.start_training, font=tkf.Font(size=12))
        self.modelStatus: tk.Label = tk.Label(self, bg="white", fg="red", font=("Arial", 10))
        # Pack widgets
        self.title.pack()
        self.about.pack()
        self.theoryButton.pack()
        self.trainButton.pack()
        self.modelStatus.pack()
        # Setup
        self.pack_propagate(False)

    def manage_predicting(self, predictThread: threading.Thread):
        "Wait for model predicting thread to finish, then output prediction results"
        if not predictThread.is_alive():
            # Output example prediction results
            plt.imshow(self.catModel.testInputs[:,0].reshape((64,64,3)))
            print("Cat" if np.squeeze(self.catModel.prediction)[0] >= 0.5 else "Not a cat")
            plt.show()
            plt.imshow(self.catModel.testInputs[:,14].reshape((64,64,3)))
            print("Cat" if np.squeeze(self.catModel.prediction)[14] >= 0.5 else "Not a cat")
            plt.show()
        else:
            self.after(1_000, self.manage_predicting, predictThread)

    def manage_training(self, trainThread: threading.Thread):
        "Wait for model training thread to finish, then start predicting with model in new thread"
        if not trainThread.is_alive():
            # Plot losses of model training
            plt.plot(np.squeeze(self.catModel.trainLosses))
            plt.xlabel("Epochs")
            plt.ylabel("Loss Value")
            plt.show()
            # Start predicting thread
            self.modelStatus.configure(text="Using trained weights and bias to predict", fg="green")
            predictThread: threading.Thread = threading.Thread(target=self.catModel.predict)
            predictThread.start()
            self.manage_predicting(predictThread)
        else:
            self.after(1_000, self.manage_training, trainThread)

    def start_training(self):
        "Start training model in new thread"
        self.modelStatus.configure(text="training weights and bias...")
        trainThread: threading.Thread = threading.Thread(target=self.catModel.train, args=(5_000,))
        trainThread.start()
        self.manage_training(trainThread)