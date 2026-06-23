# for loading data

import gzip
import pickle

import numpy as np

def load_data() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray]
    ]:
    """
    It returns a tuple containg the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry is a numpy ndarray with 50,000 entries. Each entry 
    is, in turn, a numpy ndarray with 784 values, representing the 
    28 * 28 = 784 pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries. Those entries are just the digit values
    (0...9) for the corresponding images contained in the first entry 
    of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    """

    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f)
    return (training_data, validation_data, test_data)

def vectorised_result(j: int) -> np.ndarray:
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def load_data_wrapper() -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, int]],
    list[tuple[np.ndarray, int]]
    ]:
    """
    ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (781,1)) for x in tr_d[0]]
    training_results = [vectorised_result(y) for y in tr_d[1]]
    # zip dosen't create a list in python3
    training_data = list(zip(training_inputs,training_results))
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)