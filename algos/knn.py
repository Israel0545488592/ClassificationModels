import numpy as np
from numpy.linalg import norm


class Knn:

    ''' Knn classifire '''

    def __init__(self, data: np.ndarray, labels: np.ndarray, k: int, metric: str) -> None:

        self.data = data
        self.labels = labels
        self.k = k

        if   metric == 'l1':    self.distance = lambda x, y : sum(abs(x - y))
        elif metric == 'l2':    self.distance = lambda x, y : norm(x - y)
        elif metric == 'linf':  self.distance = lambda x, y : max(abs(x - y))

        else: raise ValueError('metric paramter can be l1, l2 or linf')
        
 
    def classify(self, sample: np.ndarray) -> float:
        
        knns = sorted(zip(self.data, self.labels), key = lambda dot : self.distance(dot[0], sample))[ : self.k]

        return True if sum([neighbour[1] for neighbour in knns]) > self.k // 2 else False

    
    def test(self, data: np.ndarray, labels: np.ndarray):
        return sum(self.classify(instance) == label for instance, label in zip(data, labels)) / len(data)