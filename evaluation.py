import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import scikitplot as skplt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)
random.seed(1001)

category = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",

            "Residential", "River", "SeaLake"]


"""
Loading the Test Data
"""
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

"""
Pixel Normalization
"""
X_test = X_test / 255

model = load_model("model.h5")


"""
Prediction of probability and classes of test data. 
"""
prediction_probability = model.predict_proba(X_test, verbose=0)
prediction_class = model.predict_classes(X_test, verbose=0)


"""
Classification report includes precision, recall, and F1 score.
"""
accuracy = accuracy_score(y_test, prediction_class)
print('Accuracy Score: %f' % accuracy)

classification_report = classification_report(y_test, prediction_class, target_names=category)
print(classification_report)


"""
Area Under the Receiver Operating Characteristic Curve (ROC/AUC).
"""

roc_auc = roc_auc_score(y_test, prediction_probability, average="macro", multi_class="ovo")
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC): %f' % roc_auc)

plot_roc = skplt.metrics.plot_roc(y_test, prediction_probability)
plot_roc.figure.savefig("ROC")

"""
Confusion Matrix
"""

confusion_matrix = confusion_matrix(y_test, prediction_class)
print("Confusion Matrix:")
print(confusion_matrix)

plot_CM = skplt.metrics.plot_confusion_matrix(y_test, prediction_class)
plot_CM.figure.savefig("Confusion_Matrix.jpg")
