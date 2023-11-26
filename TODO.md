### Todo

- More tests? (Add asserts?)
- Make UI nicer ~ (use customtkinter?)

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

- Maybe save optimum models
- Maybe give option to continue training pretrained model
- Maybe add ability to add external .npz file

- Update setup.py version

- Jupyter notebook for technical presentation?

- Note that Cat dataset has 50 test images (fewer needed for testing) and MNIST has 10,000
- Note that tkinter manage method is an example of recursion for report?
- Note: Mention time complexity in report + speed times of saving + loading, why cat acc low ?