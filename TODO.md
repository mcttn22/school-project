### Todo

- Add encapsulation
- Test save model on GPU

- Finish analysis draft
  - Complete analysis interview
  - Talk about more comparison (parameters, datasets etc)
  - Update Matrice theory
  - Format images to correct sections
  - Submit to Teams

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

- Start design draft
  - Update Class diagram (load_model frame etc)

- Allow weights to be saved in files with sqlite3? (save numpy arrays in an array to a file with np.save, then store the location of the file and the model parameters under an ID with sqlite3) (user gives name to model)
  - Load frame for dataset (chosen on homepage) -> select name from drop down menu, when selected shows all parameters
  - Generate and export methods in model class
  - Pretrained optimum? - Leave for now

- Maybe give option to continue training pretrained model

- Update setup.py version

- Jupyter notebook for technical presentation?

- Note that Cat dataset has 50 test images (fewer needed for testing) and MNIST has 10,000