### Todo

- Write project-report Analysis
  - About
    - Project focus
      - Investigation into Image recognition from scratch
        - No 3rd party machine learning libraries
      - Where have I researched about it (IBM?)
      - Who is it for -> Interview with person in AI
    - What deep learning is (explains what project is investigating)
  
  - Objectives:
    - Learn how Neural Networks work and build them from scratch
    - Train models on datasets
      - Use GPU on desktop
      - Store weights with sqlite3
    - Setup remote access to training
      - Server on desktop
      - Webpage to control and display results

  - Modelling project (theory behind it)
    - Theory behind Neural Networks
      - Structure
        - Layers + neurons (shapes of networks (note overfitting))
          - Diagram
        - Step by step through layers to get prediction
          - Matrices
            - dot product
            - transposing
          - Activation function
          - Currently only sigmoid transfer function but can try to add more
      - Type of Neural Networks made
        - Supervised (binary vs multi class classification)
        - Feed-Forward
        - Uses Logistice regression (outputs probability of event a etc)
      - How they learn
        - Forward Propagation
          - The process of feeding inputs in and getting a prediction
        - Back Propagations
          - Calculating loss (type of loss function used)
          - Gradient descent
            - Multivariable => multidimensional => non euclidian
            - False minima
            - Exploding gradients
            - Calculus (chain rule, multivariable calculus)
            - Jacobian matrice
        - Deep Model theory (maybe perceptron if doesn't work on cat recognition and keeping cat recognition)
      - How datasets are used
        - XOR gate
        - MNIST (multi class (note source))
        - Cat recognition (note source)
  - Theory behind using GPU to train
    - Faster computation time needed as training takes long
    - How GPU works / why good for this
      - Cuda, Tensor~
  - Theory behind storing weights with sqlite3
  - Theory behind networking
    - Networking Diagrams
  - Theory behind webpage
    - UI diagrams
  - Remove 'docs/'

- Setup Deep Model for cat recognintion
  - Setup default values
  - If deep cat recognition doesn't work, just leave as perceptron or remove?

- Try using GPUs for calculations
  - Note dependency for CUDA toolkit, Nvidia graphics card of x+ gen, GPU drivers etc
  - If decide not to use networking, then maybe give option to just use CPU to train?

- Setup networking
  - Setup server on desktop (Flask?) (Use public ip?)
    - Returns webpage with controls to train weights + predict on desktop's GPUs
      - Store trained weights in files with sqlite3 etc
  - Try giving option for more types of transfer functions (then can compare if wanted)
  - left=options, right=train,cancel training, training progress, epoch count control

- Add prediction correctness to MNIST model?