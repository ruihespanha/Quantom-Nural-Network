import numpy as np
import pandas as pd
import seaborn as sns
import ImageLoader as im
from qiskit import transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from sklearn.decomposition import PCA
from qiskit.quantum_info.operators import Operator
from qiskit.providers.basic_provider import BasicProvider
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

path = "C:\\Users\\ruihe\\Quantom-Nural-Network\\ALL DATA"

# extracts images
x_raw, y_raw = im.ImageLoader(path, 100)

# class0, class1, class2 = 0, 1, 2

# # Select all the images that belong to the selected classes
# ix = np.where((y_raw == class0) | (y_raw == class1) | (y_raw == class2))
# x_raw, y_raw = x_raw[ix], y_raw[ix]

# # We consider only a sub-sample of 200 observations (100 per class)
# # that will be splitted into training and test set

# # seed to reproduce the results
# seed = 123
# np.random.seed(seed)

# # number of observation for each class
# n = 100

# # Generate a new dataset composed by only 100 observations
# # for each class of interest
# mask = np.hstack(
#     [
#         np.random.choice(np.where(y_raw == l)[0], n, replace=False)
#         for l in np.unique(y_raw)
#     ]
# )
# np.random.shuffle(mask)
# x_raw, y_raw = x_raw[mask], y_raw[mask]

# # The size in percentage of data the training set
# train_size = 0.90  # 200 x 0.9 = 180 training observations

# # Random splitting of dataset in training and test
# num_data = len(y_raw)
# num_train = int(train_size * num_data)
# index = np.random.permutation(range(num_data))

# # Training set
# X_train = x_raw[index[:num_train]]
# Y_train = y_raw[index[:num_train]]

# # Test set
# X_test = x_raw[index[num_train:]]
# Y_test = y_raw[index[num_train:]]

# # The variable ncol stores the total number of pixels to represent the images
# ncol = x_raw.shape[1] * x_raw.shape[2]

# # We construct the dataset where each row represents an image each column a pixel
# x_flat = X_train.reshape(-1, ncol)  # (180, 784)

# print(x_flat.shape)
# # We have 180 images in the training set described by 784 pixels

# # Rename the columns
# feat_cols = ["pixel" + str(i) for i in range(x_flat.shape[1])]

# # construction of the pandas dataframe
# df_flat = pd.DataFrame(x_flat, columns=feat_cols)
# df_flat["Y"] = Y_train


# # From sklearn.decomposition we import the class PCA that allows performing the Principal Component Analysis

# # Two principal components are considered
# pca = PCA(n_components=12)

# # Application of the PCA to the dataset
# principalComponents = pca.fit_transform(x_flat)
# print("The size of the new dataset (no label) is :", principalComponents.shape)

# print(sum(pca.explained_variance_ratio_))

# data_pca = pd.DataFrame(
#     data=principalComponents,
#     columns=["Component " + str(i + 1) for i in range(principalComponents.shape[1])],
# )

# # Append the target variable to the new dataset
# data_pca["Y"] = df_flat.iloc[:, -1:].to_numpy()

# # Visualise the first 5 rows of the new dataset
# print(data_pca.head())

# # scatter plot of the new representation
# cols = [
#     "Component 1",
#     "Component 2",
#     "Component 3",
#     "Component 4",
#     "Component 5",
#     "Component 6",
#     "Component 7",
#     "Component 8",
#     "Component 9",
#     "Component 10",
#     "Component 11",
#     "Component 12",
# ]

# if True:
#     fig, axs = plt.subplots(3, 2)
#     axs[0, 0] = sns.jointplot(
#         x=cols[0], y=cols[1], hue="Y", data=data_pca, legend="full"
#     )
#     axs[1, 0] = sns.jointplot(
#         x=cols[2], y=cols[3], hue="Y", data=data_pca, legend="full"
#     )
#     axs[2, 0] = sns.jointplot(
#         x=cols[4], y=cols[5], hue="Y", data=data_pca, legend="full"
#     )
#     axs[0, 1] = sns.jointplot(
#         x=cols[6], y=cols[7], hue="Y", data=data_pca, legend="full"
#     )
#     axs[1, 1] = sns.jointplot(
#         x=cols[8], y=cols[9], hue="Y", data=data_pca, legend="full"
#     )
#     axs[2, 1] = sns.jointplot(
#         x=cols[10], y=cols[11], hue="Y", data=data_pca, legend="full"
#     )
#     plt.show()

# # Extract the new feature as numpy array
# x_pca = data_pca[cols].to_numpy()

# MAX = np.max(x_pca)
# MIN = np.min(x_pca)

# # Rescaling of the values of the features
# X = (x_pca - MIN) / (MAX - MIN)
# Y = data_pca.Y.to_numpy()

# # pad the vectors to size 2^2 with constant values
# padding = 0.3 * np.ones((len(X), 1))
# X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

# # print(f"X_pad: {X_pad.shape}") # matrix(180, 4)

# # print(f"X_pad: {X_pad}")

# # normalize each input
# normalization = np.sqrt(np.sum(X_pad**2, -1))
# X_norm = (
#     X_pad.transpose() / normalization
# ).transpose()  # flips matrix for possible division


# def get_angles(x):
#     beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
#     beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
#     beta2 = 2 * np.arcsin(
#         np.sqrt(x[2] ** 2 + x[3] ** 2)
#         / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
#     )

#     return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# # angles for state preparation are new features
# features = np.nan_to_num((np.array([get_angles(x) for x in X_norm])))

# # number of parameters for each layer
# n_param_L = 6

# # total number of parameters
# n_parameters = n_param_L * 2 + 1

# # We define our training set, that will be the input of our QML model
# X_train = features.copy()
# Y_train = (Y - class0) / (class1 - class0)


# def state_preparation(a, circuit, target):
#     a = 2 * a
#     circuit.ry(a[0], target[0])

#     circuit.cx(target[0], target[1])
#     circuit.ry(a[1], target[1])
#     circuit.cx(target[0], target[1])
#     circuit.ry(a[2], target[1])

#     circuit.x(target[0])
#     circuit.cx(target[0], target[1])
#     circuit.ry(a[3], target[1])
#     circuit.cx(target[0], target[1])
#     circuit.ry(a[4], target[1])
#     circuit.x(target[0])

#     return circuit


# def get_Sx(ang, circuit):  # attempting to fix this func
#     simulator = AerSimulator()

#     q = QuantumRegister(2)
#     circuit = QuantumCircuit(q)
#     circuit = state_preparation(ang, circuit, [0, 1])
#     if circuit == True:
#         return circuit
#     circuit.save_unitary()

#     # job = execute(circuit, backend)
#     compiled_circuit = transpile(circuit, simulator)
#     result = simulator.run(compiled_circuit).result()

#     # print(dir(result)) print the functions of result class

#     # result = job.result()
#     # print(f"result: {type(result)}")

#     U = result.get_unitary(circuit)
#     S = Operator(U)  # not used
#     # print("Circuit unitary:\n", np.asarray(S).round(5))
#     return S


# gate = get_Sx(
#     ang=features[1], circuit=True
# )  # , x=None, pad=True, circuit=True) # ang in wrong datatype
# # print(gate.draw())

# if False:
#     gate.draw(output="mpl")
#     plt.show()


# def linear_operator(param, circuit):
#     simulator = AerSimulator()

#     data_reg = QuantumRegister(2)
#     qc = QuantumCircuit(data_reg)
#     qc.u(param[0], param[1], param[2], data_reg[0])
#     qc.u(param[3], param[4], param[5], data_reg[1])
#     qc.cx(data_reg[0], data_reg[1])
#     if circuit == True:
#         return qc
#     qc.save_unitary()

#     compiled_circuit = transpile(qc, simulator)
#     result = simulator.run(compiled_circuit).result()

#     U = result.get_unitary(qc)
#     G = Operator(U)  # not used
#     return G


# # linear_operator needs to be inisialised by defining a set of parameters
# parameters = np.repeat(np.pi, n_param_L)
# G = linear_operator(parameters, True)
# # padded variables regulate the size of the input vector. If pad=yes, a normalised four-dimensional real vector is assumed
# # the variable 'circuit=True' set out whether the function outputs the unitary matrices or the quantum circuit

# # G.draw("mpl")


# def R_gate(beta, circuit):
#     simulator = AerSimulator()
#     control = QuantumRegister(1)
#     qc = QuantumCircuit(control)
#     qc.ry(beta, control)
#     if circuit == True:
#         return qc

#     qc.save_unitary()
#     compiled_circuit = transpile(qc, simulator)
#     result = simulator.run(compiled_circuit).result()

#     U = result.get_unitary(qc)
#     R = Operator(U)

#     return R


# R = R_gate(np.pi, circuit=True)
# # R.draw("mpl")


# def sigma(circuit):
#     simulator = AerSimulator()
#     data = QuantumRegister(2)
#     qc = QuantumCircuit(data)
#     qc.id(data)
#     if circuit == True:
#         return qc

#     qc.save_unitary()

#     compiled_circuit = transpile(qc, simulator)
#     result = simulator.run(compiled_circuit).result()

#     U = result.get_unitary(qc)
#     I = Operator(U)
#     return I


# s = sigma(True)
# # s.draw("mpl")


# # The following function creates a quantum circuit that takes as
# # input the input features x and the set of parameters


# def create_circuit_compact(parameters=None, x=None, pad=True):

#     # Total number of parameters in the quantum circuit
#     n_params = len(parameters)

#     # Parameters of quantum gates for control, data and temp register
#     beta = parameters[0]
#     theta1 = parameters[1 : int((n_params + 1) / 2)]
#     theta2 = parameters[int((n_params + 1) / 2) : int(n_params)]

#     # Initialization of the quantum circuit
#     control = QuantumRegister(1, "control")
#     data = QuantumRegister(2, "data")
#     temp = QuantumRegister(2, "temp")
#     c = ClassicalRegister(1)
#     qc = QuantumCircuit(control, data, temp, c)

#     ### STATE PREPARATION

#     # Encode data into a quantum state
#     S = get_Sx(ang=x, circuit=False)
#     print(np.asarray(S).round(5))
#     print(S)

#     qc.unitary(S, data, label="$S_{x}$")  # problum with S

#     # Initialisation of the control qubit
#     R = R_gate(beta, circuit=False)
#     qc.unitary(R, control, label="$R_{Y}(β)$")

#     qc.barrier()

#     ### LINEAR TRANSFORMATIONS IN SUPERPOSITION

#     # cswap between data and temp register
#     qc.cswap(control, data[0], temp[0])
#     qc.cswap(control, data[1], temp[1])

#     # Apply quantum gate G(ϴ) to the data register
#     G1 = linear_operator(theta1, circuit=False)
#     qc.unitary(G1, data, label="$G(θ_{1})$")

#     # Apply quantum gate G(ϴ) to the temp register
#     G2 = linear_operator(theta2, circuit=False)
#     qc.unitary(G2, temp, label="$G(θ_{2})$")

#     # cswap between data and temp register
#     qc.cswap(control, data[1], temp[1])
#     qc.cswap(control, data[0], temp[0])

#     qc.barrier()

#     ### SINGLE EXECUTION OF THE ACTIVATION FUNCTION

#     sig = sigma(circuit=False)
#     qc.unitary(sig, data, label="$Σ$")

#     qc.barrier()

#     ### MEASUREMENT

#     qc.measure(data[0], c)
#     return qc


# def create_circuit(parameters=None, x=None, pad=True):
#     n_params = len(parameters)

#     beta = parameters[0]
#     theta1 = parameters[1 : int((n_params + 1) / 2)]
#     theta2 = parameters[int((n_params + 1) / 2) : int(n_params)]

#     control = QuantumRegister(1, "control")
#     data = QuantumRegister(2, "data")
#     temp = QuantumRegister(2, "temp")
#     c = ClassicalRegister(1)
#     qc = QuantumCircuit(control, data, temp, c)
#     # print("List the qubits in this circuit:", qc.qubits)
#     # print("List the classical bits in this circuit:", qc.clbits)

#     S = get_Sx(ang=x, circuit=True)
#     R = R_gate(beta, circuit=True)
#     sig = sigma(circuit=True)
#     G1 = linear_operator(theta1, circuit=True)
#     G2 = linear_operator(theta2, circuit=True)

#     qc.compose(R.to_instruction(), qubits=control, inplace=True)  # R has no CLbits
#     qc.compose(S.to_instruction(), qubits=data, inplace=True)  # S has no CLbits

#     """
#     Type "Operator" does not have clbits. Only type "Instruction" and type "Gate" have clbits.
#     """

#     qc.barrier()
#     qc.cswap(control, data[0], temp[0])
#     qc.cswap(control, data[1], temp[1])
#     qc.barrier()

#     qc.compose(G1.to_instruction(), qubits=data, inplace=True)
#     qc.compose(G2.to_instruction(), qubits=temp, inplace=True)

#     qc.barrier()
#     qc.cswap(control, data[1], temp[1])
#     qc.cswap(control, data[0], temp[0])

#     qc.barrier()

#     qc.compose(sig.to_instruction(), qubits=data, inplace=True)
#     qc.barrier()
#     qc.measure(data[0], c)
#     return qc


# qc = create_circuit(parameters=range(n_parameters + 1), x=features[0])
# qc.draw(output="mpl")

# # qc=create_circuit(parameters=range(n_parameters+1), x=features[0])
# qc = create_circuit_compact(parameters=range(n_parameters + 1), x=features[0])
# # qc.draw(output="mpl")


# from qiskit.compiler import transpile

# qc = transpile(qc, optimization_level=3)
# # qc.draw(output="mpl")
# # plt.show()


# def execute_circuit(parameters, x=None, shots=1024, print_=False, backend=None):
#     if backend is None:
#         backend = AerSimulator()

#     qc = create_circuit(parameters, x)
#     if print_:
#         qc.draw(output="mpl")
#         plt.show()

#     compiled_circuit = transpile(qc, backend)
#     simulation = backend.run(compiled_circuit, shots=shots)
#     result = simulation.result()

#     # print(f"results: {result.results}")  # there are no results in result.results[]

#     counts = result.get_counts(qc)  # ERROR Here
#     result = np.zeros(2)
#     for key in counts:
#         result[int(key, 2)] = counts[key]
#     result /= shots
#     return result[1]


# def binary_crossentropy(labels, predictions):
#     """
#     Compare a set of predictions and the real values to compute the Binary Crossentropy for a binary target variable
#     :param labels: true values for a binary target variable.
#     :param predictions: predicted probabilities for a binary target variable
#     :return: the value of the binary cross entropy. The lower the value is, the better are the predictions.
#     """
#     loss = 0
#     for l, p in zip(labels, predictions):
#         # print(l,p)
#         loss = loss - l * np.log(np.max([p, 1e-8]))

#     loss = loss / len(labels)
#     return loss


# def cost(params, X, labels):
#     predictions = [execute_circuit(params, x) for x in X]
#     return binary_crossentropy(labels, predictions)


# # Parameter initialisation
# init_params = np.repeat(1, n_parameters)
# print(init_params)

# # Compute the prediction of the randomly intialised qSLP for the observations in the training set
# probs_train = [execute_circuit(init_params, x) for x in X_train]


# def predict(probas):
#     return (probas >= 0.5) * 1


# # Given the probabilities for the two classes, that are computed as the two basis states
# # of the first qubit of the data register, the function 'predict' computes the predicted class
# # of the target variable
# predictions_train = [predict(p) for p in probs_train]

# # Once we have the true and the predicted classes we can compute
# # the accuracy and the cross entropy (loss) of the model given the initial parametrisation


# def accuracy(labels, predictions):
#     """
#     Compare a set of predictions and the real values to compute the Accuracy for a binary target variable
#     :param labels: true values for a binary target variable.
#     :param predictions: predicted values for a binary target variable
#     :return: the value of the binary cross entropy. The lower the value is, the better are the predictions.
#     """
#     loss = 0
#     for l, p in zip(labels, predictions):
#         if abs(l - p) < 1e-5:
#             loss = loss + 1
#     loss = loss / len(labels)

#     return loss


# # accuracy
# acc_train = accuracy(Y_train, predictions_train)

# # loss
# loss = cost(init_params, X_train, Y_train)

# print("Random: | Cost: {:0.7f} | Acc train: {:0.3f}" "".format(loss, acc_train))

# batch_size = 10
# epochs = 10
# acc_final_tr = 0

# from scipy.optimize import minimize

# # define the optimiser
# # optimizer_step = COBYLA(maxiter=10, tol=0.01, disp=False)

# # define the initial set of parameters
# point = init_params

# # optimisation
# for i in range(epochs):
#     batch_index = np.random.randint(0, num_train, (batch_size,))
#     X_batch = X_train[batch_index]
#     Y_batch = Y_train[batch_index]

#     print(
#         "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.3f}"
#         "".format(i + 1, cost(point, X_train, Y_train), acc_train)
#     )

#     obj_function = lambda params: cost(params, X_batch, Y_batch)
#     # point, value, nfev = optimizer_step.minimize(len(point), obj_function, initial_point=point)
#     point = minimize(obj_function, point, method="COBYLA", options={"maxiter": 10}).x

#     # Compute predictions on train and validation set
#     probs_train = [execute_circuit(point, x) for x in X_train]
#     # probs_val = [execute_circuit(point, x) for x in X_val]

#     predictions_train = [predict(p) for p in probs_train]
#     # predictions_val = [predict(p) for p in probs_val]

#     acc_train = accuracy(Y_train, predictions_train)
#     # acc_val = accuracy(Y_val, predictions_val)

#     if acc_final_tr <= acc_train:
#         best_param = point
#         acc_final_tr = acc_train
#         # acc_final_val = acc_val
#         iteration = i
