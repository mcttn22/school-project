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
        self.trainLosses: list[float] = []
        self.prediction = None
        self.predictionAccuracy = None
        self.running: bool = True
        self.numInputs = 2
        self.numHiddenNeurons = 4
        self.numOutputNeurons = 1
        # Setup pseudo random values for weight arrays
        np.random.seed(2)
        self.hiddenWeights = np.random.rand(self.numHiddenNeurons, self.numInputs)
        self.outputWeights = np.random.rand(self.numOutputNeurons, self.numHiddenNeurons)
        self.LEARNING_RATE: float = 0.1
        self.inputs = np.array([[0,0,1,1],
                               [0,1,0,1]])
        self.outputs = np.array([[0,1,1,0]])

    def __repr__(self) -> str:
        "Read current state of model"
        return f"Number of hidden neurons: {self.numHiddenNeurons}\nHidden Weights: {self.hiddenWeights.tolist()}\nOutput Weights: {self.outputWeights.tolist()}\nLearning Rate: {self.LEARNING_RATE}"

    def init_values(self):
        "Initialise weights, bias and training losses"
        self.hiddenWeights = np.random.rand(self.numHiddenNeurons, self.numInputs)
        self.outputWeights = np.random.rand(self.numOutputNeurons, self.numHiddenNeurons)
        self.trainLosses = []

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
        hiddenOutput, self.prediction = self.forward_propagation()
        # Calculate performance of model
        self.predictionAccuracy = 100 - np.mean(np.abs(self.prediction - self.outputs)) * 100

    def train(self, epochs: int) -> None:
        "Train weights"
        self.init_values()
        for epoch in range(epochs):
            if not self.running:
                break
            hiddenOutput, prediction = self.forward_propagation()
            loss = - (1/self.inputs.shape[1]) * np.sum(self.outputs * np.log(prediction) + (1 - self.outputs) * np.log(1 - prediction))
            self.trainLosses.append(loss)
            self.back_propagation(hiddenOutput=hiddenOutput, prediction=prediction)

class Experiments(tk.Frame):
    def __init__(self, root: tk.Tk, width: int, height: int):
        super().__init__(root, width=width, height=height, bg="white")
        self.HEIGHT = height
        self.WIDTH = width
        self.root = root
        # Experiments variables
        self.xorModel = XorModel()
        # Widgets
        self.menuFrame: tk.Frame = tk.Frame(self, bg="white")
        self.title: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 20), text="Experiments")
        self.about: tk.Label = tk.Label(self.menuFrame, bg="white", font=("Arial", 14), text="For experimenting with Artificial Neural Networks, a XOR single-layer model has been used for its lesser computation time")
        self.theoryButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, text="View Theory", command=lambda: os.system("open docs/xor_model.pdf"), font=tkf.Font(size=12))
        self.trainButton: tk.Button = tk.Button(self.menuFrame, width=13, height=1, text="Train Model", command=self.start_training, font=tkf.Font(size=12))
        self.learningRateScale: tk.Scale = tk.Scale(self.menuFrame, bg="white", from_=0, to=1, resolution=0.01, orient="horizontal", label="Learning Rate", length=185)
        self.learningRateScale.set(0.1)
        self.numHiddenNeuronsScale: tk.Scale = tk.Scale(self.menuFrame, bg="white", from_=1, to=20, orient="horizontal", label="Number of Hidden Neurons", length=185)
        self.numHiddenNeuronsScale.set(4)
        self.modelStatus: tk.Label = tk.Label(self.menuFrame, bg="white", fg="red", font=("Arial", 15))
        self.resultsFrame: tk.Frame = tk.Frame(self, bg="white")
        self.lossFigure: Figure = Figure()
        self.lossCanvas: FigureCanvasTkAgg = FigureCanvasTkAgg(figure=self.lossFigure, master=self.resultsFrame)
        self.results: tk.Label = tk.Label(self.resultsFrame, bg="white", font=("Arial", 20))
        # Pack widgets
        self.title.grid(row=0, column=0, columnspan=4)
        self.about.grid(row=1, column=0, columnspan=4, pady=(10,0))
        self.theoryButton.grid(row=2, column=0, pady=(10,0))
        self.trainButton.grid(row=2, column=3, pady=(10,0))
        self.learningRateScale.grid(row=3, column=1, pady=(10,0), sticky='e', padx=(0,5))
        self.numHiddenNeuronsScale.grid(row=3, column=2, pady=(10,0), sticky='w', padx=(5,0))
        self.modelStatus.grid(row=4, column=0, columnspan=4, pady=(10,0))
        self.menuFrame.pack()
        self.resultsFrame.pack(pady=(50,0))
        # Setup
        self.grid_propagate(False)
        self.pack_propagate(False)

    def manage_predicting(self, predictThread: threading.Thread):
        "Wait for model predicting thread to finish, then output prediction results"
        if not predictThread.is_alive():
            # Output example prediction results
            results = f"Prediction Accuracy: {round(self.xorModel.predictionAccuracy)}%\nNumber of Hidden Neurons: {self.xorModel.numHiddenNeurons}\n"
            for i in range(self.xorModel.inputs.shape[1]):
                results += f"{self.xorModel.inputs[0][i]},{self.xorModel.inputs[1][i]} = {1 if np.squeeze(self.xorModel.prediction)[i] >= 0.5 else 0}\n"
            self.results.configure(text=results)
            self.results.pack(side="right")
            # Enable train button
            self.trainButton["state"] = "normal"
        else:
            self.after(1_000, self.manage_predicting, predictThread)

    def manage_training(self, trainThread: threading.Thread):
        "Wait for model training thread to finish, then start predicting with model in new thread"
        if not trainThread.is_alive():
            # Plot losses of model training
            graph = self.lossFigure.add_subplot(111)
            graph.set_title(f"Learning rate: {self.xorModel.LEARNING_RATE}")
            graph.set_xlabel("Epochs")
            graph.set_ylabel("Loss Value")
            graph.plot(np.squeeze(self.xorModel.trainLosses))
            self.lossCanvas.get_tk_widget().pack(side="left")
            # Start predicting thread
            self.modelStatus.configure(text="Using trained weights to predict", fg="green")
            predictThread: threading.Thread = threading.Thread(target=self.xorModel.predict)
            predictThread.start()
            self.manage_predicting(predictThread)
        else:
            self.after(1_000, self.manage_training, trainThread)

    def start_training(self):
        "Start training model in new thread"
        # Disable train button
        self.trainButton["state"] = "disabled"
        # Reset canvases and figures
        self.lossFigure: Figure = Figure()
        self.lossCanvas.get_tk_widget().destroy()
        self.lossCanvas: FigureCanvasTkAgg = FigureCanvasTkAgg(figure=self.lossFigure, master=self.resultsFrame)
        self.results.pack_forget()
        # Start training thread
        self.xorModel.LEARNING_RATE = self.learningRateScale.get()
        self.xorModel.numHiddenNeurons = self.numHiddenNeuronsScale.get()
        self.modelStatus.configure(text="Training weights...")
        trainThread: threading.Thread = threading.Thread(target=self.xorModel.train, args=(50_000,))
        trainThread.start()
        self.manage_training(trainThread)