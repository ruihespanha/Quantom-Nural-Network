import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Image size
IMG_SIZE = 64


folder = R"C:\Users\ruihe\Quantom Nural Network\Training_Data" + "\\"
print(folder)


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            try:
                img = (
                    Image.open(f"{folder}{filename}")
                    .convert("L")
                    .resize((IMG_SIZE, IMG_SIZE))
                )
                images.append(np.array(img) / 255.0)
                labels.append(label)
            except:
                print(f"Error loading image: {folder} , {filename}")
    return images, labels


training_images, training_Labels = load_images_from_folder(folder, "images")
plt.imshow(training_images[0], interpolation="nearest")
plt.show()
