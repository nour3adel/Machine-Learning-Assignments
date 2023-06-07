#importing Libraries
import os
#opencv
import cv2
#hog
from skimage.feature import hog
#svm 
from sklearn import svm
#accuracy
from sklearn.metrics import accuracy_score
#plotting
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

##########################################################################################################

# Define the paths of the train and test folders
training_dataset = 'dataset/train/'
test_dataset = 'dataset/test/'

##########################################################################################################

# Define a function to extract the HOG features and visualize the HOG image
def feature_extraction(image_Loc): 
    image = cv2.imread(image_Loc)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (128, 64))
    fd, hog_image = hog(resized_image, orientations = 9 , pixels_per_cell = (8, 8) , cells_per_block = (2, 2) , visualize=True)
    return fd, hog_image

##########################################################################################################

categories = ['accordian', 'dollar_bill', 'motorbike', 'soccer_ball']
    
##########################################################################################################
# Extract the HOG features and visualize the HOG image of the training set
train_features = []
train_labels = []


for category in categories:   
    for IMG_NAME in os.listdir(os.path.join(training_dataset, category)):
        imageLOC = os.path.join(training_dataset, category, IMG_NAME)
        Features, hog_image = feature_extraction(imageLOC)
        train_features.append(Features)
        train_labels.append(category)

##########################################################################################################
#Plot all train set
        
fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(20, 10))
for i in range(4):
    for j in range(14):
        index = i * 5 + j
        image_name = os.listdir(os.path.join(training_dataset, categories[i]))[j]
        image_path = os.path.join(training_dataset, categories[i], image_name)
        features, hog_image = feature_extraction(image_path)
        axs[i,j].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        axs[i,j].set_title(categories[i])
        axs[i,j].axis('off')
plt.tight_layout()
plt.show()

##########################################################################################################
#Plott all hog images of train set

fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(20, 10))
for i in range(4):
    for j in range(14):
        index = i * 5 + j
        image_name = os.listdir(os.path.join(training_dataset, categories[i]))[j]
        image_path = os.path.join(training_dataset, categories[i], image_name)
        features, hog_image = feature_extraction(image_path)
        axs[i,j].imshow(hog_image)
        axs[i,j].set_title(categories[i])
        axs[i,j].axis('off')
plt.tight_layout()
plt.show()

##########################################################################################################

# Extract the HOG features and visualize the HOG image of the test set
test_features = []
test_labels = []
test_images = []

for category in categories:
    for image_name in os.listdir(os.path.join(test_dataset, category)):
        image_path = os.path.join(test_dataset, category, image_name)
        features, hog_image = feature_extraction(image_path)
        test_features.append(features)
        test_labels.append(category)
        test_images.append(cv2.imread(image_path))     

##########################################################################################################
   
# Train the SVM model
svm_model = svm.SVC(kernel='linear', C=1)
svm_model.fit(train_features, train_labels)

# Train the SVM model
# svm_model = svm.SVC(kernel='poly', degree=3)
# svm_model.fit(train_features, train_labels)


##########################################################################################################

# Test the SVM model and calculate the accuracy
testpredicted_labels = svm_model.predict(test_features)
testaccuracy = accuracy_score(test_labels, testpredicted_labels)
print("-----------------------------------")
print("Test Accuracy:", testaccuracy)
print("-----------------------------------")


trainpredicted_labels = svm_model.predict(train_features)
trainaccuracy = accuracy_score(train_labels, trainpredicted_labels)
print("-----------------------------------")
print("Train Accuracy:", trainaccuracy)
print("-----------------------------------")


##########################################################################################################

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    predicted_label = testpredicted_labels[i]
    actual_label = test_labels[i]
    if predicted_label == actual_label:
        ax.set_title(predicted_label, color='green')
    else:
        ax.set_title("Predicted: {}\nActual: {}".format(predicted_label, actual_label), color='red')
plt.show()
    
##########################################################################################################
