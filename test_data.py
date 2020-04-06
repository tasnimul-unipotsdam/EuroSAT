import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm

random.seed(1001)


category = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",
            "Residential", "River", "SeaLake"]

test_dir = "/home/abir/Documents/EuroSAT/test"


def test_data():
    data = []

    for item in category:
        test_path = os.path.join(test_dir, item)
        label = category.index(item)
        """AnnualCrop: 0, Forest: 1, HerbaceousVegetation: 2, , Highway: 3, Industrial: 4, Pasture: 5, PermanentCrop: 6,
                    Residential: 7, River: 8, SeaLake: 9"""
        for images in tqdm(os.listdir(test_path)):
            image = cv2.imread(os.path.join(test_path, images), cv2.IMREAD_COLOR)
            data.append([image, label])

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int16())
    np.save("X_test.npy", x)
    np.save("y_test.npy", y)

    return x, y


X_test, y_test = test_data()

print(X_test.shape)
print(y_test.shape)
print(X_test.dtype)
print(y_test.dtype)
print("The dataset contains {} images".format(len(y_test)))

'''
To check the dataset is not shuffled.
'''
print(y_test[:])

'''
Plotting a validation image
'''
img = X_test[14]
img = img[:, :, 0]
plt.imshow(img)
plt.show()
