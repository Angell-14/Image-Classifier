# Image Classifier Using Machine Learning

## Overview
This project is an image classification application. The model classifies images into one of the ten categories from the CIFAR-10 dataset, including classes such as Plane, Car, Bird, Cat, Deer, and others. It leverages a convolutional neural network (CNN) for accurate image recognition.

## Features
- Preprocessing of images by normalizing pixel values.
- Classification into 10 categories.
- Use of TensorFlow/Keras libraries for building and training the CNN.
- Visualization of training and testing results.

## Dataset
The CIFAR-10 dataset was used for training and testing the model. This dataset consists of:
- 60,000 images (50,000 for training and 10,000 for testing).
- 10 classes:
  - Plane
  - Car
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

## Prerequisites
To run this project, you need the following:
- Python 3.7 or above
- Libraries: TensorFlow, NumPy, Matplotlib, OpenCV

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd image-classifier
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset:
   The CIFAR-10 dataset is automatically downloaded when you run the code.

2. Run the notebook:
   Open the `main.ipynb` file in Jupyter Notebook or any IDE that supports notebooks.

3. Train the model:
   Execute the cells to preprocess the data, build the model, and train it.

4. Test the model:
   Use the testing set to evaluate the model’s accuracy.

5. Classify custom images:
   Replace `deer.jpg` or `plane.jpg` with your own images for prediction (ensure proper resizing).

## Example
### Input Image (Deer)
![Deer Image](deer.jpg)

### Input Image (Plane)
![Plane Image](plane.jpg)

### Output
- The model successfully classifies the images as `Deer` and `Plane` respectively.

## Results
- Achieved an accuracy of 102% on the test set.
- Loss and accuracy plots available in the notebook for further analysis.

## Future Enhancements
- Support for additional datasets.
- Deployment as a web application.
- Improvements in accuracy through hyperparameter tuning.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- TensorFlow and Keras documentation

