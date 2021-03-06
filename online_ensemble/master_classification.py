import numpy as np
import matplotlib.pyplot as plt
import math
import time
from online_ensemble import OnlineEnsemble
from data_reader import DataReader


def weighted_sum(results, weights):
    sum_ = 0.0
    for idx in results.keys():
        sum_ += weights[idx] * results[idx]
    return float(sum_)


def run_classification(data_file_name="d10_k0.8_t0.1.in", target_file_name="d10_k0.8_t0.1.out", n_test_chunks=100, chunk_size=1000, n_trees=8, category='random_forest_regressor'):
    # main function
    # open the file
    directory = "data/"
    reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

    # set parameters
    dim = 10


    # initialize the random forest
    start = time.time()
    x = []
    y = []
    for i in range(chunk_size):
        data_point = reader.read_data_point()
        x.append(data_point['x'])
        yy = int(data_point['y'])
        if yy == 0:
            yy = -1
        y.append(yy)
    rf = OnlineEnsemble()
    rf.insert(n_estimators=n_trees, x=x, y=y, category=category)
    last_x_chunk = x
    last_y_chunk = y

    # start test
    w0 = {}
    for idx in rf.get_idx_list():
        w0[idx] = 1.0 / n_trees

    # set parameters
    ss = 0.002
    errors = 0.0
    cnt = 0
    error_rate = 0.0
    error_rate_vec = [0.0]
    replace_flag = 1
    normalization_flag = 1

    n_test_data = n_test_chunks * chunk_size
    for i in range(chunk_size, chunk_size + n_test_data):
        data_point = reader.read_data_point()
        x = data_point['x']
        y = int(data_point['y'])
        if y == 0:
            y = -1
        results = rf.predict_results(x)
        sum_ = weighted_sum(results, w0)
        if math.tanh(sum_) > 0:
            predict = 1
        else:
            predict = -1

        # print the result
        if predict != y:
            errors += 1.0
        cnt += 1
        error_rate = errors / cnt
        if i % chunk_size == 0 and cnt >= chunk_size:
            error_rate_vec.append(error_rate)
            print 'i = {0}, predict = {1}, target = {2}, error rate = {3:.5f}, errors={4}'.format(i, predict, y, error_rate, errors)

        # update the last chunk
        last_x_chunk.append(x)
        last_y_chunk.append(y)
        del last_x_chunk[0]
        del last_y_chunk[0]

        # update weight in the case of classification
        t = math.tanh(sum_)
        for idx in results:
            w0[idx] += results[idx] * ss * (y - t) * (1 - t**2)

        if normalization_flag == 1 and n_trees > 1:
            # normalization
            sum_ = sum(w0.values())
            for idx in w0:
                w0[idx] /= sum_

        if replace_flag == 1:
            #if i % chunk_size == 0:
            #    print w0
            # replace trees
            # delete old trees
            idx_list = []
            #threshold = 1.0 / n_trees * 0.5
            if n_trees > 1:
                threshold = 1.0 / n_trees * (0.3 / (n_trees * 2 - 3) + 0.4)
            else:
                threshold = 0.5
            for idx in w0:
                if w0[idx] < threshold:
                    idx_list.append(idx)
            if len(idx_list) == 0:
                continue
            for idx in idx_list:
                del w0[idx]
            rf.delete(idx_list)
            print 'replace {0} trees whose indices are {1}'.format(len(idx_list), idx_list)
            # insert new trees
            rf.insert(len(idx_list), last_x_chunk, last_y_chunk, category=category)
            for idx in rf.get_idx_list():
                if not (idx in w0.keys()):
                    w0[idx] = 1.0 / n_trees

        if normalization_flag == 1 and n_trees > 1:
            # normalization
            sum_ = sum(w0.values())
            for idx in w0:
                w0[idx] /= sum_

    """
    print 'error_rate = ', error_rate
    plt.plot(error_rate_vec)
    plt.ylabel('error_rate')
    plt.show()
    """
    reader.close()
    return error_rate, error_rate_vec