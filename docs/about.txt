𝗔𝗯𝗼𝘂𝘁 𝗔𝗜:
• Artificial Intelligence (AI) mimics human cognition in order to perform tasks and learn from them
• Machine Learning is a subfield of AI that uses algorithms trained on data to produce adaptable models (trained programs)
• Deep Learning is a subfield of Machine Learning that uses Artificial Neural Networks (ANNs)
• The focus of this project will be on learning ANNs (creating a simple ANN from scratch) and their applications (image recognition)

𝙃𝙤𝙬 𝘼𝙧𝙩𝙞𝙛𝙞𝙘𝙞𝙖𝙡 𝙉𝙚𝙪𝙧𝙖𝙡 𝙉𝙚𝙩𝙬𝙤𝙧𝙠𝙨 𝙬𝙤𝙧𝙠:
𝙎𝙩𝙧𝙪𝙘𝙩𝙪𝙧𝙚 𝙤𝙛 𝙖𝙣 𝘼𝙉𝙉:
• Input layer of multiple inputs (in an array)
• Hidden layers of calculations (the more layers, the more accurate the prediction), consisting of the following:
    – An Activation function:
        ∗ Calculate the dot product of the input array with a hidden weight array, then sum the result with a bias
    – A Transfer function to get an output, Eg: Sigmoid function to transform the result of the Activation function to a number between 0 and 1,
    then the result can be classified as closer to 0 or 1 (known as logistic regression),with 0 being one state and 1 being another
• Output layer to output the final result of the calculations from the hidden layers, consisting of the following:
    – An Activation function:
        ∗ Caclulate the dot product of the hidden layer output with an output weight array, then sum the result with a bias
    – A Transfer function to get an output (round output to one end of range with a meaning)

𝙃𝙤𝙬 𝘼𝙉𝙉𝙨 𝙖𝙧𝙚 𝙩𝙧𝙖𝙞𝙣𝙚𝙙:
• Forward Propagation, the process of feeding inputs into the neural network and getting a result/prediction
• Back Propagation, the process of calculating the error in the prediction then adjusting the weights and biases accordingly, consisting of the following:
    – A Cost function (used for graphs and deriving formula for gradient descent):
        ∗ Finds the average of the difference between the actual and the predicted value for each input
    – Gradient descent:
        ∗ Update the hidden/output weight arrays and bias, by subtracting the rate of change of Cost with respect to Weight, multiplied with a learning rate
        ∗ This repetitive process will continue to reduce the cost to a minimum, if the learning rate is set to an appropriate value