import numpy as np
import matplotlib.pyplot as plt
import math
import time
from online_ensemble import OnlineEnsemble
from data_reader import DataReader


def run_classification():
    # main function
    # open the file
    directory = "/Users/dengjingyu/nyu/Research/data/"
    data_file_name = "predictors1.csv"
    target_file_name = "classes1.csv"
    reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

    # set parameters
    n_trees = 100
    dim = 50
    chunk_size = 500

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
    rf.insert_with_random_forest_regressor(n_estimators=n_trees, x=x, y=y)
    last_x_chunk = x
    last_y_chunk = y

    # start test
    w0 = {}
    for idx in rf.get_idx_list():
        w0[idx] = 1.0 / n_trees

    # set parameters
    ss = 0.0001
    errors = 0.0
    error_rate = 0.0
    error_rate_vec = []
    replace_flag = 1
    normalization_flag = 1

    n_test_data = 20000
    for i in range(chunk_size, chunk_size + n_test_data):
        data_point = reader.read_data_point()
        x = data_point['x']
        y = data_point['y']
        if y == 0:
            y = -1
        predict = rf.predict_weighted_classification_result(x, w0)

        # print the result
        if predict != y:
            errors += 1.0
        error_rate = errors / (i + 1 - chunk_size)
        error_rate_vec.append(error_rate)
        print 'i = {0}\tpredict = {1}\ttarget = {2}\terror rate = {3:.5f}'.format(i, predict, y, error_rate)

        # update the last chunk
        last_x_chunk.append(x)
        last_y_chunk.append(y)
        del last_x_chunk[0]
        del last_y_chunk[0]

        """
        # update weight in the case of classification
        results = rf.predict_results(x)
        sum_ = rf.predict_weighted_sum(x, w0)
        t = math.tanh(sum_)
        for idx in results:
            w0[idx] += results[idx] * ss * (y - t) * (1 - t**2)
        """
        if replace_flag == 1:
            # replace trees
            # delete old trees
            idx_list = []
            threshold = 1.0 / n_trees * 0.5
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
            rf.insert_with_random_forest_regressor(len(idx_list), last_x_chunk, last_y_chunk)
            for idx in rf.get_idx_list():
                if not (idx in w0.keys()):
                    w0[idx] = 1.0 / n_trees
        if normalization_flag == 1:
            # normalization
            sum_ = sum(w0.values())
            for idx in w0:
                w0[idx] /= sum_
    print 'error_rate = ', error_rate
    plt.plot(error_rate_vec)
    plt.ylabel('error_rate')
    plt.show()

run_classification()