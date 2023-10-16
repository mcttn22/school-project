### Todo

- Finish analysis draft
  - Complete analysis interview
  - Talk about more comparison (parameters, datasets etc)
  - Update Matrice theory
  - Format images to correct sections
  - Submit to Teams

- Fix Cat recognition Perceptron Network overflow errors

- Update MakeFile
- Remove 'docs/'

- Allow weights to be saved in files with sqlite3? (save numpy arrays in an array to a file with np.save, then store the location of the file and the model parameters under an ID with sqlite3) (user gives name to model)
  - Pretrained optimum?

- Update GUI
  - Add loading datasets / model info to UI?
  - See ui design image on phone X
  - Options over model parameters
    - Transfer functions
    - Network shape
    - Epoch count
  - Cancel training button?
  - Training progress display
  - Comparison of models
  - left=options, right=train
  - Same page structure, but add load button, then train (continue with loaded weights) vs test button
  - Add -> Load new object (have to for new load_dataset method), with initialisation parameters (Size of training data (for load_dataset method), ANN type (Perceptron, Deep with relu + shape, validate option + how often + more)) -> Test -> Results -> Save with name (Stop training button)
  - Load -> Show name options -> Choose option -> Test model -> Show results, option to delete custom ?

- Ideas from interview
  - Allow model to be saved
  - Talk about vanishing gradient (and ReLu vs sigmoid for this)
  - Validation set (can use test-dataset for now)
    - Monitors how well network is predicting, to see when network becomes overtrained
  - Browse through more of results
    - Show images predicted correct
    - Show images predicted wrong
    - Show images where prediction is close between muliple classes
  - Play around with size of training datasets
    - Graph showing performance against increase in training data
  - Add new focuses to analysis
  - Maybe cifar-10 dataset
  - Maybe use sections of images for more inputs
  - Maybe CNNs

- Maybe give option to continue training pretrained model

- Update README + setup.py version

- Jupyter notebook for technical presentation?

- Note that Cat dataset has 50 test images (fewer needed for testing) and MNIST has 10,000