import sys
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image


def mat_to_jpg(mat_path, output_jpg_path):
    # Load .mat file
    mat_data = loadmat(mat_path)
    print("All data from MAT file:")
    for key in mat_data:
        if not key.startswith("__"):
            print(
                f"{key}: {type(mat_data[key])}, shape: {getattr(mat_data[key], 'shape', None)}"
            )

    # Try to find an image-like array (2D or 3D, numeric)
    image_array = None
    for key, value in mat_data.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            if value.ndim == 2 or (value.ndim == 3 and value.shape[2] in [1, 3, 4]):
                image_array = value
                print(f"Using '{key}' as image data.")
                break

    if image_array is None:
        print("No suitable image array found in the MAT file.")
        return

    # Normalize and convert to uint8
    arr = image_array
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr *= 255
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 1:
        img = Image.fromarray(arr.squeeze(), mode="L")
    else:
        print("Unsupported image array shape.")
        return

    img.save(output_jpg_path)
    print(f"Image saved to {output_jpg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python MAT_To_JPG.py input.mat output.jpg")
        sys.exit(1)
    mat_to_jpg(sys.argv[1], sys.argv[2])
