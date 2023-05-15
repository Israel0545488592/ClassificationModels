import numpy as np

class Winnow:

    def __init__(self) -> None:
        self.weights : np.ndarray.astype(int)


    def predict(self, vector: np.ndarray) -> bool:

        if self.weights is None:                  raise RuntimeError('Model has not been trained yet')
        if len(self.weights) != len(vector):      raise RuntimeError('Instance is of the wrong format')

        return self.weights @ vector.astype(int) >= len(self.weights) // 2


    def train(self, data : np.ndarray, labels: np.ndarray) -> int:

        self.weights = np.ones(data.shape[1]).astype(int)
        mistakes = 0

        for vector, label in zip(data, labels):

                classification = self.predict(vector)

                if label is not classification:

                    mistakes += 1
                    # False Posetive VS False negative treatment
                    self.weights[vector] *= 0 if classification else 2

        return mistakes