import h5py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import threading
import tkinter as tk
import tkinter.font as tkf

class CatModel():
    "ANN model that trains to predict if an image is a cat or not a cat"
    def __init__(self) -> None:
        "Initialise model values"
        self.running: bool = True
        # Load datasets
        self.trainInputs, self.trainOutputs, self.testInputs, self.testOutputs = self.load_datasets()
        self.trainLosses: list[float] = []
        self.testPrediction: np.ndarray[float] = None
        self.testPredictionAccuracy: float = None
        # Initialise weights and bias to 0/s
        self.weights: np.ndarray[float] = np.zeros(shape=(self.trainInputs.shape[0], 1))
        self.bias: float = 0
        self.LEARNING_RATE: float = 0.001

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Weights: {self.weights}\nBias: {self.bias}\nLearning Rate: {self.LEARNING_RATE}"
    
    def init_model_values(self) -> None:
        "Initialise weights and bias"
        self.weights = np.zeros(shape=(self.trainInputs.shape[0], 1))
        self.bias = 0
    
    def load_datasets(self) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        "Returns image input arrays and output arrays for training and testing"
        # Load datasets, h5 file stores large amount of data with quick access
        trainDataset: h5py.File = h5py.File('school_project/datasets/train-cat.h5', "r")
        testDataset: h5py.File = h5py.File('school_project/datasets/test-cat.h5', "r")
        # Input arrays, containing the RGB values for each pixel in each 64x64 pixel image, for 209 images
        trainInputs: np.ndarray[float] = np.array(trainDataset["train_set_x"][:])
        testInputs: np.ndarray[float] = np.array(testDataset["test_set_x"][:])
        # Output arrays, 1 for cat, 0 for not cat
        trainOutputs: np.ndarray[float] = np.array(trainDataset["train_set_y"][:])
        testOutputs: np.ndarray[float] = np.array(testDataset["test_set_y"][:])
        # Reshape input arrays into 1 dimension (flatten), then divide by 255 (RGB) to standardize them to a number between 0 and 1
        trainInputs = trainInputs.reshape((trainInputs.shape[0], -1)).T / 255
        testInputs = testInputs.reshape((testInputs.shape[0], -1)).T / 255
        # Reshape output arrays into a 1 dimensional list of outputs
        trainOutputs = trainOutputs.reshape((1, trainOutputs.shape[0]))
        testOutputs = testOutputs.reshape((1, testOutputs.shape[0]))
        return trainInputs, trainOutputs, testInputs, testOutputs

    def sigmoid(self, z: any) -> any:
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, prediction: np.ndarray[float]) -> None:
        "Adjust the weights and bias via gradient descent"
        weightGradient: np.ndarray[float] = np.dot(self.trainInputs, (prediction - self.trainOutputs).T) / self.trainInputs.shape[1]
        biasGradient: np.ndarray[float] = np.sum(prediction - self.trainOutputs) / self.trainInputs.shape[1]
        # Update weights and bias
        self.weights -= self.LEARNING_RATE * weightGradient
        self.bias -= self.LEARNING_RATE * biasGradient

    def forward_propagation(self) -> np.ndarray[float]:
        "Generate a prediction with the weights and bias, returns a prediction"
        z1: np.ndarray[float] = np.dot(self.weights.T, self.trainInputs) + self.bias
        prediction: np.ndarray[float] = self.sigmoid(z1)
        return prediction

    def predict(self) -> None:
        "Use trained weights and bias to predict if image is a cat or not a cat"
        # Calculate prediction for test dataset
        z1: np.ndarray[float] = np.dot(self.weights.T, self.testInputs) + self.bias
        self.testPrediction = self.sigmoid(z1)
        # Calculate performance of model
        self.testPredictionAccuracy = 100 - np.mean(np.abs(self.testPrediction.round() - self.testOutputs)) * 100

    def train(self, epochs: int) -> None:
        "Train weights and bias"
        self.trainLosses = []
        for epoch in range(epochs):
            if not self.running:
                break
            prediction = self.forward_propagation()
            loss: float = - (1/self.trainInputs.shape[1]) * np.sum(self.trainOutputs * np.log(prediction) + (1 - self.trainOutputs) * np.log(1 - prediction))
            self.trainLosses.append(np.squeeze(loss))
            self.back_propagation(prediction=prediction)

class ImageRecognitionFrame(tk.Frame):
    "Frame for image recognition page"
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        "Initialise image recognition frame widgets"
        super().__init__(root, width=width, height=height, bg="white")
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        # Image recognition frame variables
        self.catModel: CatModel = CatModel()
        # Widgets
        self.menuFrame: tk.Frame = tk.Frame(self, bg="white")
        self.titleLabel: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 20), text="Image Recognition")
        self.aboutLabel: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 14), text="An Image model trained on recognising if an image is a cat or not")
        self.theoryButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, font=tkf.Font(size=12), text="View Theory", command=lambda: os.system("open docs/image_model.pdf"))
        self.trainButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, font=tkf.Font(size=12), text="Train Model", command=self.start_training)
        self.learningRateScale: tk.Scale = tk.Scale(self.menuFrame, bg="white", orient="horizontal", label="Learning Rate", length=185, from_=0, to=0.037, resolution=0.001)
        self.learningRateScale.set(self.catModel.LEARNING_RATE)
        self.modelStatusLabel: tk.Label = tk.Label(self.menuFrame, bg="white", fg="red", font=("Arial", 15))
        self.resultsFrame: tk.Frame = tk.Frame(self, bg="white")
        self.lossFigure: Figure = Figure()
        self.lossCanvas: FigureCanvasTkAgg = FigureCanvasTkAgg(figure=self.lossFigure, master=self.resultsFrame)
        self.imageFigure: Figure = Figure()
        self.imageCanvas: FigureCanvasTkAgg = FigureCanvasTkAgg(figure=self.imageFigure, master=self.resultsFrame)
        # Pack widgets
        self.titleLabel.grid(row=0, column=0, columnspan=2)
        self.aboutLabel.grid(row=1, column=0, columnspan=2, pady=(10,0))
        self.theoryButton.grid(row=2, column=0, pady=(10,0))
        self.trainButton.grid(row=2, column=1, pady=(10,0))
        self.learningRateScale.grid(row=3, column=0, columnspan=2, pady=(10,0))
        self.modelStatusLabel.grid(row=4, column=0, columnspan=2, pady=(10,0))
        self.menuFrame.pack()
        self.resultsFrame.pack(pady=(50,0))
        # Setup
        self.grid_propagate(False)
        self.pack_propagate(False)

    def manage_predicting(self, predictThread: threading.Thread) -> None:
        "Wait for model predicting thread to finish, then output prediction results"
        if not predictThread.is_alive():
            # Output example prediction results
            self.imageFigure.suptitle(f"Prediction Correctness: {round(self.catModel.testPredictionAccuracy)}%")
            image1: Figure.axes = self.imageFigure.add_subplot(121)
            image1.set_title("Cat" if np.squeeze(self.catModel.testPrediction)[0] >= 0.5 else "Not a cat")
            image1.imshow(self.catModel.testInputs[:,0].reshape((64,64,3)))
            image2: Figure.axes = self.imageFigure.add_subplot(122)
            image2.set_title("Cat" if np.squeeze(self.catModel.testPrediction)[14] >= 0.5 else "Not a cat")
            image2.imshow(self.catModel.testInputs[:,14].reshape((64,64,3)))
            self.imageCanvas.get_tk_widget().pack(side="right")
            # Enable train button
            self.trainButton["state"] = "normal"
        else:
            self.after(1_000, self.manage_predicting, predictThread)

    def manage_training(self, trainThread: threading.Thread) -> None:
        "Wait for model training thread to finish, then start predicting with model in new thread"
        if not trainThread.is_alive():
            # Plot losses of model training
            graph: Figure.axes = self.lossFigure.add_subplot(111)
            graph.set_title(f"Learning rate: {self.catModel.LEARNING_RATE}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.catModel.trainLosses))
            self.lossCanvas.get_tk_widget().pack(side="left")
            # Start predicting thread
            self.modelStatusLabel.configure(text="Using trained weights and bias to predict", fg="green")
            predictThread: threading.Thread = threading.Thread(target=self.catModel.predict)
            predictThread.start()
            self.manage_predicting(predictThread=predictThread)
        else:
            self.after(1_000, self.manage_training, trainThread)

    def start_training(self) -> None:
        "Start training model in new thread"
        # Disable train button
        self.trainButton["state"] = "disabled"
        # Reset canvases and figures
        self.lossFigure = Figure()
        self.lossCanvas.get_tk_widget().destroy()
        self.lossCanvas = FigureCanvasTkAgg(figure=self.lossFigure, master=self.resultsFrame)
        self.imageFigure = Figure()
        self.imageCanvas.get_tk_widget().destroy()
        self.imageCanvas = FigureCanvasTkAgg(figure=self.imageFigure, master=self.resultsFrame)
        # Start training thread
        self.catModel.LEARNING_RATE = self.learningRateScale.get()
        self.catModel.init_model_values()
        self.modelStatusLabel.configure(text="Training weights and bias...")
        trainThread: threading.Thread = threading.Thread(target=self.catModel.train, args=(5_000,))
        trainThread.start()
        self.manage_training(trainThread=trainThread)