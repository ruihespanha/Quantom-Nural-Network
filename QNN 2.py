# load MNIST dataset
from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.providers.basic_provider import BasicProvider
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info.operators import Operator

(x_raw, y_raw), _ = mnist.load_data()

print("Data features shape:", x_raw.shape)
print("Data labels shape:", y_raw.shape)

# plot images
num_row = 2
num_col = 5
# num_row x num_col gives the total number of images that will be plotted

# Classes of interest. The QML model will be trained to distinguisg between 0 and 9
class0, class1 = 0, 9

# Select all the images that belong to the selected classes
ix = np.where((y_raw == class0) | (y_raw == class1))
x_raw, y_raw = x_raw[ix], y_raw[ix]

# We consider only a sub-sample of 200 observations (100 per class)
# that will be splitted into training and test set

# seed to reproduce the results
seed = 123
np.random.seed(seed)

# number of observation for each class
n = 100

# Generate a new dataset composed by only 100 observations
# for each class of interest
mask = np.hstack(
    [
        np.random.choice(np.where(y_raw == l)[0], n, replace=False)
        for l in np.unique(y_raw)
    ]
)
np.random.shuffle(mask)
x_raw, y_raw = x_raw[mask], y_raw[mask]

# The size in percentage of data the training set
train_size = 0.90  # 200 x 0.9 = 180 training observations

# Random splitting of dataset in training and test
num_data = len(y_raw)
num_train = int(train_size * num_data)
index = np.random.permutation(range(num_data))

# Training set
X_train = x_raw[index[:num_train]]
Y_train = y_raw[index[:num_train]]

# Test set
X_test = x_raw[index[num_train:]]
Y_test = y_raw[index[num_train:]]

# The variable ncol stores the total number of pixels to represent the images
ncol = x_raw.shape[1] * x_raw.shape[2]

# We construct the dataset where each row represents an image each column a pixel
x_flat = X_train.reshape(-1, ncol)  # (180, 784)

print(x_flat.shape)
# We have 180 images in the training set described by 784 pixels

# Rename the columns
feat_cols = ["pixel" + str(i) for i in range(x_flat.shape[1])]

# construction of the pandas dataframe
df_flat = pd.DataFrame(x_flat, columns=feat_cols)
df_flat["Y"] = Y_train

# Visualise the first 5 rows of the dataset
print(df_flat.head())

# From sklearn.decomposition we import the class PCA that allows performing the Principal Component Analysis

# Two principal components are considered
pca = PCA(n_components=2)

# Application of the PCA to the dataset
principalComponents = pca.fit_transform(x_flat)
print("The size of the new dataset (no label) is :", principalComponents.shape)

print(sum(pca.explained_variance_ratio_))

data_pca = pd.DataFrame(
    data=principalComponents,
    columns=["Component " + str(i + 1) for i in range(principalComponents.shape[1])],
)

# Append the target variable to the new dataset
data_pca["Y"] = df_flat.iloc[:, -1:].to_numpy()

# Visualise the first 5 rows of the new dataset
print(data_pca.head())

# scatter plot of the new representation
cols = ["Component 1", "Component 2"]
if False:
    ax = sns.scatterplot(x=cols[0], y=cols[1], hue="Y", data=data_pca, legend="full")
    plt.show()

# Extract the new feature as numpy array
x_pca = data_pca[cols].to_numpy()

MAX = np.max(x_pca)
MIN = np.min(x_pca)

# Rescaling of the values of the features
X = (x_pca - MIN) / (MAX - MIN)
Y = data_pca.Y.to_numpy()

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

# print(f"X_pad: {X_pad.shape}") # matrix(180, 4)

# print(f"X_pad: {X_pad}")

# normalize each input
normalization = np.sqrt(np.sum(X_pad**2, -1))
X_norm = (
    X_pad.transpose() / normalization
).transpose()  # flips matrix for possible division


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# angles for state preparation are new features
features = np.nan_to_num((np.array([get_angles(x) for x in X_norm])))

# number of parameters for each layer
n_param_L = 6

# total number of parameters
n_parameters = n_param_L * 2 + 1

# We define our training set, that will be the input of our QML model
X_train = features.copy()
Y_train = (Y - class0) / (class1 - class0)


def state_preparation(a, circuit, target):
    a = 2 * a
    circuit.ry(a[0], target[0])

    circuit.cx(target[0], target[1])
    circuit.ry(a[1], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[2], target[1])

    circuit.x(target[0])
    circuit.cx(target[0], target[1])
    circuit.ry(a[3], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[4], target[1])
    circuit.x(target[0])

    return circuit


def get_Sx(ang):  # attempting to fix this func
    simulator = AerSimulator()

    q = QuantumRegister(2)
    circuit = QuantumCircuit(q)
    circuit = state_preparation(ang, circuit, [0, 1])
    circuit.save_unitary()

    # job = execute(circuit, backend)
    compiled_circuit = transpile(circuit, simulator)
    simulation = simulator.run(compiled_circuit, shots=2048)
    result = simulation.result()

    print(dir(result))

    # result = job.result()
    # print(f"result: {type(result)}")

    U = result.get_unitary(circuit, 6)
    S = Operator(U)
    return S


gate = get_Sx(
    ang=features[1]
)  # , x=None, pad=True, circuit=True) # ang in wrong datatype
# gate.draw("mpl") #doesnt work
