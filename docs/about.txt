# About
- Artificial Intelligence (AI) mimics human cognition in order to perform tasks and learn from them
- Machine Learning is a subfield of AI that uses algorithms trained on data to produce adaptable models (trained programs)
- Deep Learning is a subfield of Machine Learning that uses Artificial Neural Networks (ANNs)
- The focus of this project will be on learning ANNs (creating a simple ANN from scratch) and their applications (image recognition)

# How Artificial Neural Networks work
- ANNs start with forward propagation, consisting of an input layer (of multiple inputs), 
  hidden layers of nodes (using an activation function to sum all inputs multiplied with a weight and summed with a bias, then use the transfer function to get an output (sigmoid for 0 to 1 (binary))) 
  and an output layer (to output the final result of the calculations from the hidden layer),
  then after getting an output, by using backpropagation you calculate the cost of the prediction (how wrong it is) with a cost function (takes the average difference between actual and predicted values squared (to avoid negatives cancelling out)),
  then adjust the weights and bias' with the gradient of the cost function (gradient descent) to minimise the value of the cost function
  then repeat with each set of inputs