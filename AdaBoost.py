import numpy as np
from typing import Tuple


def orgenize_data(data: np.ndarray) -> Tuple[np.ndarray]:

    np.random.shuffle(data)

    train = data[: len(data) // 2]
    test = data[len(data) // 2 :]

    # splitting from labels
    return train[:, :-1], train[:, -1].astype(bool), test[:, :-1], test[:, -1].astype(bool)


if __name__ == '__main__':

    Data = np.loadtxt('squares.txt')

    train_data, train_labels, test_data, test_labels = orgenize_data(Data)

    print(list(zip(train_data, train_labels)))
    print('---------')
    print(list(zip(test_data, test_labels)))