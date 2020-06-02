from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import utils
from modules.lenet import LeNet
import glob
import numpy as np
import cv2
import os
import imutils
from config import dataset, loss_fn, opt, batchSize, num_epochs, save_model_path
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

data = []
labels = []

print("[INFO] Loading the data...")
for image_path in glob.glob(dataset):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # Extract the labels from the path
    label = image_path.split(os.path.sep)[-3]
    # rename positive to smiling and negative to not smiling
    label = "smiling" if label == "positives" else "not_smiling"
    # print(label)
    labels.append(label)


# normalize the raw pixels to the range [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Encode the labels
le = LabelEncoder()
labels = utils.to_categorical(le.fit_transform(labels), 2)

# Handle the data imbalance by computing the class weight
total_classes = labels.sum(axis=0)
# print("total classes: ", total_classes)
class_weights = total_classes.max() / total_classes

class_weights = {i: class_weights[i] for i in range(1,-1,-1)}
print("class weights: ", class_weights)

# class_weights = {
#     0: 1.,
#     1: 2.56
# }

# train-test split
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=5)
print("train shape: ",train_x[0].shape)
print("[INFO] Compiling the model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])
print("Real shape: ", train_x[0].shape)
print("[INFO] Training the network...")
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weights, batch_size=batchSize, epochs=num_epochs, verbose=1)

print("[INFO] Evaluating the model...")
preds = model.predict(test_x, batch_size=batchSize)
print(classification_report(test_y.argmax(axis=1), preds.argmax(axis=1), target_names=le.classes_))

print("[INFO] Saving the model...")
model.save(save_model_path)

# Visualize

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), history.history["loss"], label="training loss")
plt.plot(np.arange(0,15), history.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,15), history.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,15), history.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
