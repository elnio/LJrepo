from data_reader import DataReader
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def flatten(l):
    ret = []
    for i in range(len(l)):
        ret = ret + l[i]
    return ret


def run_classification(data_file_name="d10_k0.8_t0.1.in", target_file_name="d10_k0.8_t0.1.out", n_test_chunks=100):
    directory = "data/"
    reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

    chunk_size = 1000
    n_classifiers = 8  # number of classifiers

    errors = 0.0
    cnt = 0
    nc = 0
    error_rate_vec = []

    ri = 0
    x_chunks = []
    y_chunks = []
    for i in range(n_classifiers):
        x, y = reader.read_data_chunk(chunk_size)
        x_chunks.append(x)
        y_chunks.append(y)
    clf = DecisionTreeClassifier()
    clf.fit(flatten(x_chunks), flatten(y_chunks))


    for i in range(n_test_chunks):
        # read data chunk
        x, y = reader.read_data_chunk(chunk_size)

        # test previous classifiers
        for k in range(chunk_size):
            xk = x[k]
            yk = y[k]
            res = clf.predict(xk)
            if res != yk:
                errors += 1
            cnt += 1
        nc += 1
        error_rate = errors / cnt
        error_rate_vec.append(error_rate)
        print 'tested {0} chunks {1} data points, error_rate = {2:.5f}, errors = {3}'.format(i + 1,
                                                                                      cnt,
                                                                                      errors / cnt,
                                                                                      errors)

        # train a new classifier
        x_chunks[ri] = x
        y_chunks[ri] = y
        ri = (ri + 1) % n_classifiers
        clf.fit(flatten(x_chunks), flatten(y_chunks))
    """
    print 'error_rate = ', error_rate
    plt.plot(error_rate_vec)
    plt.ylabel('error_rate')
    plt.show()
    """
    reader.close()
    return error_rate_vec