import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm

random.seed(1001)

category = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",
            "Residential", "River", "SeaLake"]

train_dir = "/home/abir/Documents/EuroSAT/train"


def train_data():
    data = []

    for item in category:
        train_path = os.path.join(train_dir, item)
        label = category.index(item)
        """AnnualCrop: 0, Forest: 1, HerbaceousVegetation: 2, , Highway: 3, Industrial: 4, Pasture: 5, PermanentCrop: 6,
                    Residential: 7, River: 8, SeaLake: 9"""

        for images in tqdm(os.listdir(train_path)):
            image = cv2.imread(os.path.join(train_path, images), cv2.IMREAD_COLOR)
            data.append([image, label])
    data = random.sample(data, len(data))

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int16())
    np.save("X_train.npy", x)
    np.save("y_train.npy", y)

    return x, y


X_train, y_train = train_data()

print(X_train.shape)
print(y_train.shape)
print(X_train.dtype)
print(y_train.dtype)
print("The dataset contains {} images".format(len(y_train)))

'''
To check the dataset is shuffled.
'''
print(y_train[:20])

'''
Plotting a validation image
'''
img = X_train[14]
img = img[:, :, 0]
plt.imshow(img)
plt.show()
