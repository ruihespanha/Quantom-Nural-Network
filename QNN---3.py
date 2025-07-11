import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
from scipy.optimize import minimize

# Image size
IMG_SIZE = 64


def load_images_from_folder(folder, label):
    count = 0
    images = []
    labels = []
    for filename in os.listdir(folder):
        # if count == 20:
        #     break
        # count += 1
        if filename.lower().endswith((".jpg", ".png")):
            try:
                img = (
                    Image.open(os.path.join(folder, filename))
                    .convert("L")
                    .resize((IMG_SIZE, IMG_SIZE))
                )
                images.append(np.array(img) / 255.0)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image: {filename}", e)
    return images, labels


# base folder containing subfolders Glioma, Meningioma, Pituitary
base = r"C:\Users\ruihe\Quantom Nural Network\ALL DATA"
glioma_images, glioma_labels = load_images_from_folder(os.path.join(base, "Glioma"), 0)
meningioma_images, meningioma_labels = load_images_from_folder(
    os.path.join(base, "Meningioma"), 1
)
pituitary_images, pituitary_labels = load_images_from_folder(
    os.path.join(base, "Pituitary"), 2
)

# combine into single arrays
x_raw = np.stack(
    glioma_images + meningioma_images + pituitary_images
)  # shape (ΣN,64,64)
y_raw = np.array(glioma_labels + meningioma_labels + pituitary_labels)  # shape (ΣN,)

print("Data features shape:", x_raw.shape)
print("Data labels shape:  ", y_raw.shape)

# stratified train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    x_raw, y_raw, test_size=0.1, random_state=123, stratify=y_raw
)
print("Train set:", X_train.shape, Y_train.shape)
print("Test set: ", X_test.shape, Y_test.shape)

# flatten images for PCA
ncol = IMG_SIZE * IMG_SIZE
x_flat = X_train.reshape(-1, ncol)  # (n_train, 4096)
print(x_flat.shape)

# Rename the columns
feat_cols = ["pixel" + str(i) for i in range(ncol)]

# construction of the pandas dataframe
df_flat = pd.DataFrame(x_flat, columns=feat_cols)
df_flat["Y"] = Y_train

# Two principal components are considered
pca = PCA(n_components=2)

# Application of the PCA to the dataset
principalComponents = pca.fit_transform(x_flat)
print("The size of the new dataset (no label) is :", principalComponents.shape)

print("Explained variance ratio sum:", sum(pca.explained_variance_ratio_))

data_pca = pd.DataFrame(
    data=principalComponents,
    columns=["Component 1", "Component 2"],
)

# Append the target variable to the new dataset
data_pca["Y"] = df_flat["Y"].to_numpy()

# Visualise the first 5 rows of the new dataset
print(data_pca.head())

# scatter plot of the new representation
plt.figure()
sns.scatterplot(x="Component 1", y="Component 2", hue="Y", data=data_pca, legend="full")
plt.title("MRI PCA embedding")
plt.show()

# Extract the new feature as numpy array
x_pca = data_pca[["Component 1", "Component 2"]].to_numpy()

MAX = np.max(x_pca)
MIN = np.min(x_pca)

# Rescaling of the values of the features
X = (x_pca - MIN) / (MAX - MIN)
Y = data_pca["Y"].to_numpy()

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

# normalize each input
normalization = np.sqrt(np.sum(X_pad**2, -1))
X_norm = (X_pad.T / normalization).T


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# angles for state preparation are new features
features = np.nan_to_num(np.array([get_angles(x) for x in X_norm]))

# number of parameters for each layer
n_param_L = 6

# total number of parameters
n_parameters = n_param_L * 2 + 1

# We define our training set, that will be the input of our QML model
# For binary QNN, select two classes (e.g., Glioma vs Meningioma)
mask = np.isin(Y_train, [0, 1])
X_train = features[mask]
Y_train = (Y_train[mask] - 0) / (1 - 0)

# Test set preparation (similarly)
# flatten X_test → PCA → pad → normalize → features → mask for classes 0 and 1 if you wish


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


def get_Sx(ang, circuit):  # attempting to fix this func
    simulator = AerSimulator()
    q = QuantumRegister(2)
    circ = QuantumCircuit(q)
    circ = state_preparation(ang, circ, [0, 1])
    if circuit is True:
        return circ
    circ.save_unitary()
    compiled_circ = transpile(circ, simulator)
    result = simulator.run(compiled_circ).result()
    U = result.get_unitary(circ)
    S = Operator(U)
    return S


def linear_operator(param, circuit):
    simulator = AerSimulator()
    data_reg = QuantumRegister(2)
    qc = QuantumCircuit(data_reg)
    qc.u(param[0], param[1], param[2], data_reg[0])
    qc.u(param[3], param[4], param[5], data_reg[1])
    qc.cx(data_reg[0], data_reg[1])
    if circuit is True:
        return qc
    qc.save_unitary()
    compiled_qc = transpile(qc, simulator)
    result = simulator.run(compiled_qc).result()
    U = result.get_unitary(qc)
    G = Operator(U)
    return G


def R_gate(beta, circuit):
    simulator = AerSimulator()
    control = QuantumRegister(1)
    qc = QuantumCircuit(control)
    qc.ry(beta, control)
    if circuit is True:
        return qc
    qc.save_unitary()
    compiled_qc = transpile(qc, simulator)
    result = simulator.run(compiled_qc).result()
    U = result.get_unitary(qc)
    R = Operator(U)
    return R


def sigma(circuit):
    simulator = AerSimulator()
    data = QuantumRegister(2)
    qc = QuantumCircuit(data)
    qc.id(data)
    if circuit is True:
        return qc
    qc.save_unitary()
    compiled_qc = transpile(qc, simulator)
    result = simulator.run(compiled_qc).result()
    U = result.get_unitary(qc)
    I = Operator(U)
    return I


def create_circuit(parameters=None, x=None, pad=True):
    n_params = len(parameters)
    beta = parameters[0]
    theta1 = parameters[1 : int((n_params + 1) / 2)]
    theta2 = parameters[int((n_params + 1) / 2) : int(n_params)]
    control = QuantumRegister(1, "control")
    data = QuantumRegister(2, "data")
    temp = QuantumRegister(2, "temp")
    c = ClassicalRegister(1)
    qc = QuantumCircuit(control, data, temp, c)

    S = get_Sx(ang=x, circuit=True)
    R = R_gate(beta, circuit=True)
    sig = sigma(circuit=True)
    G1 = linear_operator(theta1, circuit=True)
    G2 = linear_operator(theta2, circuit=True)

    qc.compose(R.to_instruction(), qubits=control, inplace=True)
    qc.compose(S.to_instruction(), qubits=data, inplace=True)
    qc.barrier()

    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])
    qc.barrier()

    qc.compose(G1.to_instruction(), qubits=data, inplace=True)
    qc.compose(G2.to_instruction(), qubits=temp, inplace=True)

    qc.barrier()
    qc.cswap(control, data[1], temp[1])
    qc.cswap(control, data[0], temp[0])
    qc.barrier()

    qc.compose(sig.to_instruction(), qubits=data, inplace=True)
    qc.barrier()
    qc.measure(data[0], c)
    return qc


def execute_circuit(parameters, x=None, shots=1024, print_=False, backend=None):
    if backend is None:
        backend = AerSimulator()
    qc = create_circuit(parameters, x)
    if print_:
        qc.draw(output="mpl")
        plt.show()
    compiled_circuit = transpile(qc, backend)
    simulation = backend.run(compiled_circuit, shots=shots)
    result = simulation.result()
    counts = result.get_counts(qc)
    outcome = np.zeros(2)
    for key in counts:
        outcome[int(key, 2)] = counts[key]
    outcome /= shots
    return outcome[1]


def binary_crossentropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss -= l * np.log(np.max([p, 1e-8]))
    return loss / len(labels)


def cost(params, X, labels):
    predictions = [execute_circuit(params, x) for x in X]
    return binary_crossentropy(labels, predictions)


def predict(probas):
    return (probas >= 0.5) * 1


def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss += 1
    return loss / len(labels)


# Parameter initialisation
init_params = np.repeat(1, n_parameters)
print(init_params)

# Compute the prediction of the randomly intialised qSLP for the observations in the training set
probs_train = [execute_circuit(init_params, x) for x in X_train]
predictions_train = [predict(p) for p in probs_train]

# accuracy
acc_train = accuracy(Y_train, predictions_train)
# loss
loss = cost(init_params, X_train, Y_train)
print(f"Random: | Cost: {loss:0.7f} | Acc train: {acc_train:0.3f}")

batch_size = 10
epochs = 10
acc_final_tr = 0
point = init_params

for i in range(epochs):
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_batch = X_train[batch_index]
    Y_batch = Y_train[batch_index]

    print(
        f"Iter: {i+1:5d} | Cost: {cost(point, X_train, Y_train):0.7f} | Acc train: {acc_train:0.3f}"
    )

    obj_function = lambda params: cost(params, X_batch, Y_batch)
    point = minimize(obj_function, point, method="COBYLA", options={"maxiter": 10}).x

    probs_train = [execute_circuit(point, x) for x in X_train]
    predictions_train = [predict(p) for p in probs_train]
    acc_train = accuracy(Y_train, predictions_train)

    if acc_final_tr <= acc_train:
        best_param = point
        acc_final_tr = acc_train
        iteration = i
