# About AI
- Artificial Intelligence (AI) mimics human cognition in order to perform tasks and learn from them
- Machine Learning is a subfield of AI that uses algorithms trained on data to produce adaptable models (trained programs)
- Deep Learning is a subfield of Machine Learning that uses Artificial Neural Networks (ANNs)
- The focus of this project will be on learning ANNs (creating a simple ANN from scratch) and their applications (image recognition)

# How Artificial Neural Networks work
## Structure of an ANN
- Input layer of multiple inputs (in a matrice)
- Hidden layers of calculations, consisting of the following:
  - An Activation function:
    - Multiply each input with a weight then sum the result with a bias
    - Output the summation of all of the results
  - A Transfer function to get an output, Eg: Sigmoid function to transform the result of the Activation function to a number between 0 and 1
- Output layer to output the final result of the calculations from the hidden layers
## How ANNs are trained
- Forward Propagation, the process of feeding inputs into the neural network and getting a result/prediction
- Back Propagation, the process of calculating the error in the prediction then adjusting the weights and biases accordingly, consisting of the following:
  - A Cost function:
    - Finds the average of the difference between the actual and the predicted values squared, this is squared to stop negatives cancelling out
  - Gradient descent:
    - Update the weight and bias, by subtracting the gradient of the cost function, at the point of the cost value calculated, multiplied with a learning rate
    - This repetitive process will continue to reduce the cost to a minimum, if the learning rate is set to an appropriate value