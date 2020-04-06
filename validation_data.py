import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm

random.seed(1001)


category = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",
            "Residential", "River", "SeaLake"]

validation_dir = "/home/abir/Documents/EuroSAT/validation"


def validation_data():
    data = []

    for item in category:
        validation_path = os.path.join(validation_dir, item)
        label = category.index(item)
        """AnnualCrop: 0, Forest: 1, HerbaceousVegetation: 2, , Highway: 3, Industrial: 4, Pasture: 5, PermanentCrop: 6,
                    Residential: 7, River: 8, SeaLake: 9"""

        for images in tqdm(os.listdir(validation_path)):
            image = cv2.imread(os.path.join(validation_path, images), cv2.IMREAD_COLOR)
            data.append([image, label])

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int16())
    np.save("X_validation.npy", x)
    np.save("y_validation.npy", y)

    return x, y


X_validation, y_validation = validation_data()

print(X_validation.shape)
print(y_validation.shape)
print(X_validation.dtype)
print(y_validation.dtype)
print("The dataset contains {} images".format(len(y_validation)))

'''
To check the validation dataset is not shuffled.
'''
print(y_validation[:])

'''
Plotting a validation image
'''
img = X_validation[14]
img = img[:, :, 0]
plt.imshow(img)
plt.show()
