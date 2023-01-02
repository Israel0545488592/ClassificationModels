import numpy as np
from typing import Tuple
from itertools import combinations


def orgenize_data(data: np.ndarray) -> Tuple[np.ndarray]:

    np.round(data, 4)
    np.random.shuffle(data)

    train = data[: len(data) // 2]
    test = data[len(data) // 2 :]

    # splitting from labels
    return train[:, :-1], train[:, -1].astype(bool), test[:, :-1], test[:, -1].astype(bool)


class AdaBoost:

    ''' The Adaptive Boosting huristic for 2D lines'''

    def __init__(self) -> None:
        self.rules = {}


    # axuliry class abstracting a rule
    class line:

        def __init__(self, x1: int, x2: int, y1: int, y2: int) -> None:

            if abs(x1 - x2) < 0.01:
                # vertical line
                self.slope = None
                self.height = x1

            else:

                self.slope = (y1 - y2) / (x1 - x2)
                self.height = y1 - self.slope * x1

        def classify(self, x: int, y: int) -> bool:
            
            return  x > self.height if self.slope is None else  y > self.slope * x + self.height

    # a way to traverse all rules
    def line_iterator(self, data: np.ndarray):

        for ind, comb in enumerate(combinations(data, 2)): yield ind, self.line(comb[0][0], comb[1][0], comb[0][1], comb[1][1])

    # error of a rule on the data
    def error(self, rule: line, data: np.ndarray, labels: np.ndarray, datas_weights: np.ndarray) -> float:

        return sum(datas_weights[ind] for ind in range(len(data)) if rule.classify(data[ind][0], data[ind][1]) != labels[ind])


    def train(self, data: np.ndarray, labels: np.ndarray, iterations: int):

        rule_weights = {}
        instances_weights = np.ones(len(data)) / len(data)

        for itr in range(iterations):
            
            ind, rule_itr = min(self.line_iterator(data), key = lambda ind_rule : self.error(ind_rule[1], data, labels, instances_weights))

            # updating chosen rule's weight
            error = self.error(rule_itr, data, labels, instances_weights)
            rule_weight = np.log((1 - error) / error) / 2
            rule_weights[ind] = rule_weight

            # updating instances weights
            for ind in range(len(instances_weights)):
                instances_weights[ind] *= np.exp(rule_weight * (-1 if rule_itr.classify(data[ind, 0], data[ind, 1]) == labels[ind] else 1))

            # normalizing
            instances_weights /= sum(instances_weights)

        # finally extracting the best eight, rules
        winners_indexes = sorted(rule_weights, key = lambda k : rule_weights[k])[-1:-9:-1]

        for ind, rule in self.line_iterator(data):

            if len(winners_indexes) == 0:   break

            if ind in winners_indexes:

                self.rules[rule] = rule_weights[ind]
                winners_indexes.remove(ind)

            else:   del rule


    def classify(self, x: int, y: int) -> bool:

        if len(self.rules) == 0: raise RuntimeError(' mpdel has not been trained yet')

        return sum(self.rules[rule] * (1 if rule.classify(x, y) else -1) for rule in self.rules) > 0


    def test(self, data: np.ndarray, labels: np.ndarray) -> float:

        return sum(self.classify(dot[0], dot[1]) == label  for dot, label in zip(data, labels)) / len(data)

if __name__ == '__main__':

    Data = np.loadtxt('squares.txt')

    train_data, train_labels, test_data, test_labels = orgenize_data(Data)

    adaboost = AdaBoost()

    adaboost.train(train_data, train_labels, 60)

    print('train score:', adaboost.test(train_data, train_labels))
    print('test score:', adaboost.test(test_data, test_labels))

    # problomes: do i update the weights of unchosen rules
    # what causes the warnning