This project demonstrates the classification of medical images into different categories using a Convolutional Neural Network (CNN). The dataset used is the Medical MNIST Dataset, which contains images from six categories: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, and HeadCT.

Project Features

Data Preprocessing: Normalization and augmentation of medical images.

Image Augmentation: Techniques like rotation, zooming, flipping, and shifting are applied to improve generalization.

CNN Architecture: A simple yet effective CNN model for image classification.

Training: Model training with augmented data.

Validation: Validation accuracy of 99.63% on 10% of the dataset.

Testing: Code to test new images using the trained model.

Dataset
https://www.kaggle.com/datasets/andrewmvd/medical-mnist

TOPICS:

ImageDataGenerator class in TensorFlow/Keras is a powerful tool for real-time data augmentation and preprocessing of image datasets. It helps enhance the generalization of machine learning models by applying various transformations to images during training.

Sequential API is a simple way to build a linear stack of layers for a neural network. It allows you to add layers one by one in a straightforward, ordered manner.

Conv2D: A convolutional layer to extract spatial features.

MaxPooling2D: A pooling layer to down-sample feature maps.

Flatten: Flattens the 2D feature maps into a 1D vector.

Dense: A fully connected layer for classification.

Dropout: A regularization layer to reduce overfitting.

Output Layer:
    Use softmax activation for multi-class classification.
    Use sigmoid activation for binary classification.
    
ReLU (Rectified Linear Unit) activation function is one of the most widely used activation functions in deep learning, especially in Convolutional Neural Networks (CNNs). It introduces non-linearity into the model while maintaining simplicity and efficiency.

Adam (Adaptive Moment Estimation) optimizer combines the benefits of both the Adagrad and RMSProp algorithms, making it robust and efficient for training deep neural networks.
     Combines the benefits of momentum and adaptive learning rates.
    Requires minimal tuning of hyperparameters.
    
categorical_crossentropy is a loss function commonly used in multi-class classification problems. It calculates the cross-entropy between the true labels and the predicted labels when the target variable is categorical, and the model outputs probabilities for each class.It is particularly useful when the output layer of your model uses a softmax activation function to predict multiple classes.
