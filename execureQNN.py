def execute_circuit(parameters, x=None, shots=1000, print=False, backend=None):
    if backend is None:
        backend = BasicAer.get_backend("qasm_simulator")

    circuit = create_circuit(parameters, x)
    if print:
        circuit.draw(output="mpl")
        plt.show()

    result = execute(circuit, backend, shots=shots).result()

    counts = result.get_counts(circuit)
    result = np.zeros(2)
    for key in counts:
        result[int(key, 2)] = counts[key]
    result /= shots
    return result[1]
