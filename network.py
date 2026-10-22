"""Dense network and explicit gradient computations using NumPy."""
from __future__ import annotations

import numpy as np


class Network:
    def __init__(self, sizes: list[int], seed: int = 0):
        if len(sizes) < 2 or any(n < 1 for n in sizes):
            raise ValueError("Need positive input/output layer sizes")
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.biases = [np.zeros((1, b)) for b in sizes[1:]]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        preactivations = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ weight + bias
            preactivations.append(z)
            activations.append(np.maximum(z, 0) if index < len(self.weights) - 1 else z)
        logits = activations[-1]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        return probabilities, activations, preactivations

    def loss_gradients(self, x: np.ndarray, labels: np.ndarray) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        probabilities, acts, pre = self.forward(x)
        if x.ndim != 2 or labels.shape != (len(x),) or len(x) == 0:
            raise ValueError("Expected nonempty batch and one label per row")
        if np.any((labels < 0) | (labels >= probabilities.shape[1])):
            raise ValueError("Label outside output classes")
        loss = -np.log(np.maximum(probabilities[np.arange(len(x)), labels], 1e-300)).mean()
        delta = probabilities.copy()
        delta[np.arange(len(x)), labels] -= 1
        delta /= len(x)
        grad_w = [np.empty_like(w) for w in self.weights]
        grad_b = [np.empty_like(b) for b in self.biases]
        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = acts[layer].T @ delta
            grad_b[layer] = delta.sum(axis=0, keepdims=True)
            if layer:
                delta = (delta @ self.weights[layer].T) * (pre[layer - 1] > 0)
        return float(loss), grad_w, grad_b

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 64,
              learning_rate: float = 0.05, momentum: float = 0.9, decay: float = 0.98, seed: int = 0) -> list[dict]:
        if epochs < 1 or batch_size < 1 or learning_rate <= 0 or not 0 <= momentum < 1 or not 0 < decay <= 1:
            raise ValueError("Invalid training parameters")
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(epochs):
            for indices in np.array_split(rng.permutation(len(x)), max(1, (len(x) + batch_size - 1) // batch_size)):
                _, gw, gb = self.loss_gradients(x[indices], y[indices])
                for j in range(len(self.weights)):
                    self.velocity_w[j] = momentum * self.velocity_w[j] - learning_rate * gw[j]
                    self.velocity_b[j] = momentum * self.velocity_b[j] - learning_rate * gb[j]
                    self.weights[j] += self.velocity_w[j]
                    self.biases[j] += self.velocity_b[j]
            loss, _, _ = self.loss_gradients(x, y)
            accuracy = float(np.mean(self.predict(x) == y))
            history.append({"epoch": epoch + 1, "loss": loss, "accuracy": accuracy})
            learning_rate *= decay
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x)[0], axis=1)


def gradient_error(model: Network, x: np.ndarray, labels: np.ndarray, epsilon: float = 1e-5) -> float:
    _, gw, gb = model.loss_gradients(x, labels)
    errors = []
    for arrays, gradients in ((model.weights, gw), (model.biases, gb)):
        for array, gradient in zip(arrays, gradients):
            for index in np.ndindex(array.shape):
                original = array[index]
                array[index] = original + epsilon
                plus = model.loss_gradients(x, labels)[0]
                array[index] = original - epsilon
                minus = model.loss_gradients(x, labels)[0]
                array[index] = original
                numeric = (plus - minus) / (2 * epsilon)
                errors.append(abs(numeric - gradient[index]) / max(1e-12, abs(numeric) + abs(gradient[index])))
    return max(errors)

