import numpy as np


class Winnow:

    def __init__(self) -> None:
        
        self.weights : np.ndarray.astype(int)


    def predict(self, vector: np.ndarray) -> bool:


        if self.weights is None:                  raise RuntimeError('Model has not been trained yet')
        if len(self.weights) != len(vector):      raise RuntimeError('Instance is of the wrong format')

        return self.weights @ vector.astype(int) >= len(self.weights) // 2


    def train(self, src : str) -> int:

        # loading data
        arr = np.loadtxt(src).astype(bool)
        # organizing
        data = arr[:, :-1]
        labels = arr[:, -1]
        del arr

        ''' Executing the algorithm '''

        self.weights = np.ones(data.shape[1]).astype(int)
        mistakes = 0

        while True:

            for vector, label in zip(data, labels):

                classification = self.predict(vector)

                if label is not classification:

                    mistakes += 1
                    # False Posetive
                    if classification:  self.weights[vector] = 0
                    # False Negative
                    else:               self.weights[vector] *= 2

                    continue

            return mistakes



if __name__ == '__main__':

    winnow = Winnow()
    print('mistake count:', winnow.train('winnow_vectors.txt'))
    print('final weights:', winnow.weights)