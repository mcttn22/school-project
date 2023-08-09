from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import threading
import tkinter as tk
import tkinter.font as tkf

class XorModel():
    "ANN model that trains to predict the output of a XOR gate with two inputs"
    def __init__(self) -> None:
        "Initialise model values"
        self.running: bool = True
        # Setup model data
        self.trainInputs: np.ndarray[float] = np.array([[0,0,1,1],
                                                 [0,1,0,1]])
        self.trainOutputs: np.ndarray[float] = np.array([[0,1,1,0]])
        self.trainLosses: list[float] = []
        self.testPrediction: np.ndarray[float] = None
        self.testPredictionAccuracy: float = None
        # Model attributes
        self.numInputs: int = self.trainInputs.shape[0]
        self.numHiddenNeurons: int = 4
        self.numOutputNeurons: int = 1
        # Initialise weights to random values
        ## Setup pseudo random values for weight arrays
        np.random.seed(2)
        self.hiddenWeights: np.ndarray[float] = np.random.rand(self.numHiddenNeurons, self.numInputs)
        self.outputWeights: np.ndarray[float] = np.random.rand(self.numOutputNeurons, self.numHiddenNeurons)
        self.LEARNING_RATE: float = 0.1

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Number of hidden neurons: {self.numHiddenNeurons}\nHidden Weights: {self.hiddenWeights.tolist()}\nOutput Weights: {self.outputWeights.tolist()}\nLearning Rate: {self.LEARNING_RATE}"

    def init_weights(self) -> None:
        "Initialise weights to randdom values"
        self.hiddenWeights = np.random.rand(self.numHiddenNeurons, self.numInputs)
        self.outputWeights = np.random.rand(self.numOutputNeurons, self.numHiddenNeurons)

    def sigmoid(self, z: any) -> any:
        "Transfer function, transforms input to number between 0 and 1"
        return 1 / (1 + np.exp(-z))

    def back_propagation(self, hiddenOutput: np.ndarray[float], testPrediction: np.ndarray[float]) -> None:
        "Adjust the weights via gradient descent"
        outputWeightGradient: np.ndarray[float] = np.dot(testPrediction - self.trainOutputs, hiddenOutput.T) / self.trainInputs.shape[1]
        hiddenWeightGradient: np.ndarray[float] = np.dot(np.dot(self.outputWeights.T, testPrediction - self.trainOutputs) * hiddenOutput * (1 - hiddenOutput), self.trainInputs.T) / self.trainInputs.shape[1]
        # Reshape arrays to match the weight arrays for multiplication
        outputWeightGradient = np.reshape(outputWeightGradient, self.outputWeights.shape)
        hiddenWeightGradient = np.reshape(hiddenWeightGradient, self.hiddenWeights.shape)
        # Update weights
        self.hiddenWeights -= self.LEARNING_RATE * hiddenWeightGradient
        self.outputWeights -= self.LEARNING_RATE * outputWeightGradient

    def forward_propagation(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        "Generate a prediction with the weights, returns the hidden layer output and a prediction"
        z1: np.ndarray[float] = np.dot(self.hiddenWeights, self.trainInputs)
        hiddenOutput: np.ndarray[float] = self.sigmoid(z1)
        z2: np.ndarray[float] = np.dot(self.outputWeights, hiddenOutput)
        testPrediction: np.ndarray[float] = self.sigmoid(z2)
        return hiddenOutput, testPrediction

    def predict(self) -> None:
        "Use trained weights to predict ouput of XOR gate on two inputs"
        hiddenOutput, self.testPrediction = self.forward_propagation()
        # Calculate performance of model
        self.testPredictionAccuracy = 100 - np.mean(np.abs(self.testPrediction - self.trainOutputs)) * 100

    def train(self, epochs: int) -> None:
        "Train weights"
        self.trainLosses = []
        for epoch in range(epochs):
            if not self.running:
                break
            hiddenOutput, testPrediction = self.forward_propagation()
            loss: float = - (1/self.trainInputs.shape[1]) * np.sum(self.trainOutputs * np.log(testPrediction) + (1 - self.trainOutputs) * np.log(1 - testPrediction))
            self.trainLosses.append(loss)
            self.back_propagation(hiddenOutput=hiddenOutput, testPrediction=testPrediction)

class ExperimentsFrame(tk.Frame):
    "Frame for experiments page"
    def __init__(self, root: tk.Tk, width: int, height: int) -> None:
        "Initialise experiments frame widgets"
        super().__init__(root, width=width, height=height, bg="white")
        self.root = root
        self.WIDTH = width
        self.HEIGHT = height
        # Experiments variables
        self.xorModel = XorModel()
        # Widgets
        self.menuFrame: tk.Frame = tk.Frame(self, bg="white")
        self.titleLabel: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 20), text="Experiments")
        self.aboutLabel: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 14), text="For experimenting with Artificial Neural Networks, a XOR single-layer model has been used for its lesser computation time")
        self.theoryButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, font=tkf.Font(size=12), text="View Theory", command=lambda: os.system("open docs/xor_model.pdf"))
        self.trainButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, font=tkf.Font(size=12), text="Train Model", command=self.start_training)
        self.learningRateScale: tk.Scale = tk.Scale(self.menuFrame, bg="white", orient="horizontal", label="Learning Rate", length=185, from_=0, to=1, resolution=0.01)
        self.learningRateScale.set(self.xorModel.LEARNING_RATE)
        self.numHiddenNeuronsScale: tk.Scale = tk.Scale(self.menuFrame, bg="white", orient="horizontal", label="Number of Hidden Neurons", length=185, from_=1, to=20, resolution=1.0)
        self.numHiddenNeuronsScale.set(self.xorModel.numHiddenNeurons)
        self.modelStatusLabel: tk.Label = tk.Label(self.menuFrame, bg="white", fg="red", font=("Arial", 15))
        self.resultsFrame: tk.Frame = tk.Frame(self, bg="white")
        self.lossFigure: Figure = Figure()
        self.lossCanvas: FigureCanvasTkAgg = FigureCanvasTkAgg(figure=self.lossFigure, master=self.resultsFrame)
        self.results: tk.Label = tk.Label(self.resultsFrame, bg="white", font=("Arial", 20))
        # Pack widgets
        self.titleLabel.grid(row=0, column=0, columnspan=4)
        self.aboutLabel.grid(row=1, column=0, columnspan=4, pady=(10,0))
        self.theoryButton.grid(row=2, column=0, pady=(10,0))
        self.trainButton.grid(row=2, column=3, pady=(10,0))
        self.learningRateScale.grid(row=3, column=1, padx=(0,5), pady=(10,0), sticky='e')
        self.numHiddenNeuronsScale.grid(row=3, column=2, padx=(5,0), pady=(10,0), sticky='w')
        self.modelStatusLabel.grid(row=4, column=0, columnspan=4, pady=(10,0))
        self.menuFrame.pack()
        self.resultsFrame.pack(pady=(50,0))
        # Setup
        self.grid_propagate(False)
        self.pack_propagate(False)

    def manage_predicting(self, predictThread: threading.Thread) -> None:
        "Wait for model predicting thread to finish, then output testPrediction results"
        if not predictThread.is_alive():
            # Output example testPrediction results
            results: str = f"Prediction Accuracy: {round(self.xorModel.testPredictionAccuracy)}%\nNumber of Hidden Neurons: {self.xorModel.numHiddenNeurons}\n"
            for i in range(self.xorModel.trainInputs.shape[1]):
                results += f"{self.xorModel.trainInputs[0][i]},{self.xorModel.trainInputs[1][i]} = {1 if np.squeeze(self.xorModel.testPrediction)[i] >= 0.5 else 0}\n"
            self.results.configure(text=results)
            self.results.pack(side="right")
            # Enable train button
            self.trainButton["state"] = "normal"
        else:
            self.after(1_000, self.manage_predicting, predictThread)

    def manage_training(self, trainThread: threading.Thread) -> None:
        "Wait for model training thread to finish, then start predicting with model in new thread"
        if not trainThread.is_alive():
            # Plot losses of model training
            graph: Figure.axes = self.lossFigure.add_subplot(111)
            graph.set_title(f"Learning rate: {self.xorModel.LEARNING_RATE}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.xorModel.trainLosses))
            self.lossCanvas.get_tk_widget().pack(side="left")
            # Start predicting thread
            self.modelStatusLabel.configure(text="Using trained weights to predict", fg="green")
            predictThread: threading.Thread = threading.Thread(target=self.xorModel.predict)
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
        self.results.pack_forget()
        # Start training thread
        self.xorModel.LEARNING_RATE = self.learningRateScale.get()
        self.xorModel.numHiddenNeurons = self.numHiddenNeuronsScale.get()
        self.xorModel.init_weights()
        self.modelStatusLabel.configure(text="Training weights...")
        trainThread: threading.Thread = threading.Thread(target=self.xorModel.train, args=(50_000,))
        trainThread.start()
        self.manage_training(trainThread=trainThread)