from data.utils import orgenize_data
from algos.winnow import Winnow
from numpy import loadtxt



if __name__ == '__main__':

    arr = loadtxt('data/winnow_vectors.txt').astype(bool)
    data = arr[:, :-1]
    labels = arr[:, -1]
    
    w = Winnow()
    mistakes = w.train(data, labels)
    print(w.weights, mistakes)