import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def ImageLoader(folder, num):
    IMG_SIZE = 640
    images = []
    view = []
    labels = []
    label_num = 0
    for sub_folder in os.listdir(folder):
        if sub_folder != ".DS_Store":
            for sub_sub_folder in os.listdir(f"{folder}" + "\\" + f"{sub_folder}"):
                if sub_sub_folder != ".DS_Store":
                    count = 0
                    for filename in os.listdir(
                        f"{folder}"
                        + "\\"
                        + f"{sub_folder}"
                        + "\\"
                        + f"{sub_sub_folder}"
                    ):
                        if filename != ".DS_Store":
                            count += 1
                            if (count >= num + 1) and (num != -1):
                                break

                            print(
                                f"{sub_folder} : {sub_sub_folder}: {filename}, {count}/{num}",
                                end="\r",
                                flush=True,
                            )
                            try:
                                img = (
                                    Image.open(
                                        f"{folder}"
                                        + "\\"
                                        + f"{sub_folder}"
                                        + "\\"
                                        + f"{sub_sub_folder}"
                                        + "\\"
                                        + f"{filename}"
                                    )
                                    .convert("L")
                                    .resize((IMG_SIZE, IMG_SIZE))
                                )
                                images.append(np.array(img) / 255.0)
                                labels.append(label_num)
                                view.append(int(sub_sub_folder))
                            except:
                                print(f"Error loading image: {folder} , {filename}")
                    label_num += 1
                print("")

    return np.stack(images), np.array(labels)
