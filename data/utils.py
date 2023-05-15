from typing import Tuple
import numpy as np

def orgenize_data(data: np.ndarray) -> Tuple[np.ndarray]:

    np.round(data, 5)
    np.random.shuffle(data)

    train = data[: len(data) // 2]
    test = data[len(data) // 2 :]

    # splitting from labels and to train test pairs
    return train[:, :-1], train[:, -1].astype(bool), test[:, :-1], test[:, -1].astype(bool)