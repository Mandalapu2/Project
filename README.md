# Classify images as radial or non-radial using CNN

This project demonstrates the use of CNN to classify archaeological disk images as radial or non-radial based on the patterns in the image.

To run the project, follow the steps provided below:

1. Install required packages
```
pip install -r requirements.txt
```

2. Run preprocessing code to create new dataset with splitted images
```
python image_preprocessing.py
```
Once the code runs successfully, you should find a new folder named "cropped" inside the data directory. This is the new dataset that will be used to train our neural network.

3. Run classification code to train the model
```
python radial_vs_non_radial.py
```
While this code is running, you should be able to see the training and validation score for each epoch of the 7 folds. The average score is also printed at the end.

The scripts are also available in jupyter notebook. You can find two .ipynb files:
1. image_preprocessing.ipynb: To create new dataset
2. radial_vs_non_radial.ipynb: To train the model
