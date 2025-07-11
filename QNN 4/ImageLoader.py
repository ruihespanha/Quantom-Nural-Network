import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def ImageLoader(folder, num):
    IMG_SIZE = 640
    images = []
    labels = []
    label_num = 0
    for sub_folder in os.listdir(folder):
        count = 0

        for filename in os.listdir(f"{folder}" + "\\" + f"{sub_folder}"):
            count += 1
            if (count >= num + 1) and (num != -1):
                print("hit")
                break

            print(f"{sub_folder} : {filename}, {count}/{num}", end="\r", flush=True)
            try:
                img = (
                    Image.open(
                        f"{folder}" + "\\" + f"{sub_folder}" + "\\" + f"{filename}"
                    )
                    .convert("L")
                    .resize((IMG_SIZE, IMG_SIZE))
                )
                images.append(np.array(img) / 255.0)
                labels.append(label_num)
            except:
                print(f"Error loading image: {folder} , {filename}")
        label_num += 1

    return np.stack(images), np.array(labels)
