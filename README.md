# -Emotion-Detection
Python Code for Emotion Detection

1.Imports:

The code begins by importing necessary libraries:
os for file and directory handling.
librosa for audio processing and feature extraction.
numpy for numerical computations and array handling.
pandas for data manipulation.
train_test_split from sklearn for splitting the dataset.
Keras modules from TensorFlow for building and training the neural network.
2.Load Audio Features:

The function load_audio_features takes a folder path as input, iterates through the directories, and processes audio files to extract MFCC features.
For each audio file:
The audio file is loaded using librosa.load.
MFCC features are computed, and the mean of these features is taken.
The extracted features and corresponding labels (folder names) are appended to their respective lists.
If no features are found, a ValueError is raised to notify the user.
3.Prepare Dataset:

The audio folder path is specified, and the features and labels are loaded using the load_audio_features function.
Labels are encoded as integers using a mapping dictionary.
The dataset is split into training and validation sets using an 80/20 ratio.
4.Build the Model:

A feedforward neural network is constructed using Keras' Sequential model.
The model consists of:
An input layer with 128 neurons and ReLU activation.
A dropout layer to prevent overfitting.
A hidden layer with 64 neurons and ReLU activation.
Another dropout layer.
An output layer with softmax activation for multi-class classification.
5.Compile the Model:

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function, suitable for multi-class classification tasks.
6.Train the Model:

The model is trained using the training dataset with early stopping to prevent overfitting. If the validation loss doesn't improve for 5 consecutive epochs, training will stop, and the best weights will be restored
