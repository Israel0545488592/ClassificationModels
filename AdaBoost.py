import numpy as np
from typing import Tuple
from itertools import combinations


def orgenize_data(data: np.ndarray) -> Tuple[np.ndarray]:

    np.random.shuffle(data)

    train = data[: len(data) // 2]
    test = data[len(data) // 2 :]

    # splitting from labels
    return train[:, :-1], train[:, -1].astype(bool), test[:, :-1], test[:, -1].astype(bool)


class AdaBoost:

    ''' The Adaptive Boosting huristic for 2D lines'''

    def __init__(self) -> None:
        self.weights : np.ndarray.astype(float)

    # axuliry class abstracting a rule
    class line:

        def __init__(self, x1: int, x2: int, y1: int, y2: int) -> None:
            
            self.slope = (y1 - y2) / (x1 - x2)
            self.height = y1 - self.slope * x1

        def classify(self, x: int, y: int) -> bool: return y > self.slope * x + self.height

    # a way to traverse all rules
    def line_iterator(self, data: np.ndarray):

        for ind, comb in enumerate(combinations(data, 2)):  yield ind, self.line(comb[0, 0], comb[1, 0], comb[0, 1], comb[1, 1])


    def train(self, data: np.ndarray, labels: np.ndarray, iterations: int):

        self.rule_weights = {}
        self.instances_weights = np.ones(len(data)) * (1 / len(data))

        for itr in range(iterations):
            
            ind, rule_itr = min(self.line_iterator(data), key = lambda ind, rule : self.error(rule, data))

            # updating chosen rule's weight
            error = self.error(data, rule_itr)
            rule_weight = np.log((1 - error) / error) / 2
            self.rule_weights[ind] = rule_weight

            # updating instances weights
            for ind in range(len(self.instances_weights)):
                self.instances_weights[ind] *= -rule_weight if rule_itr.classify(data[ind, 0], data[ind, 1]) else rule_weight

            # normalizing
            self.instances_weights /= sum(self.instances_weights)

    # error of a rule on the data
    def error(self, rule: line, data: np.ndarray) -> float:

        return sum(self.instances_weights[ind] for ind in range(len(data)) if rule.classify(data[ind, 0], data[ind, 1]))


if __name__ == '__main__':

    Data = np.loadtxt('squares.txt')

    train_data, train_labels, test_data, test_labels = orgenize_data(Data)