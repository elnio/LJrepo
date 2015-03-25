import numpy as np
import matplotlib.pyplot as plt
import math
import time
from online_ensemble import OnlineEnsemble
from data_reader import DataReader


def run_simple(rf, last_x_chunk, last_y_chunk, reader):
    w0 = {}
    for idx in rf.get_idx_list():
        w0[idx] = 1.0 / n_trees

    ss = 0.0001
    mse = 0.0
    mse_vec = []
    replace_flag = 1
    normalization_flag = 1

    n_test_data = 20000
    for i in range(chunk_size, chunk_size + n_test_data):
        data_point = reader.read_data_point()
        x = data_point['x']
        y = data_point['y']
        predict = rf.predict_weighted_sum(x, w0)

        # print the result
        print 'i = {0}\tpredict = {1:.5f}\ttarget = {2:.5f}\tmse = {3:.5f}'.format(i, predict, y, mse / (i + 1 - chunk_size))
        mse += (predict - y) ** 2
        mse_vec.append(mse / (i + 1 - chunk_size))

        # update the last chunk
        last_x_chunk.append(x)
        last_y_chunk.append(y)
        del last_x_chunk[0]
        del last_y_chunk[0]

        # simple version
        results = rf.predict_results(x)
        for idx in results:
            w0[idx] += results[idx] * ss * (y - predict)

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
    print 'mse = ', mse / n_test_data
    plt.plot(mse_vec)
    plt.ylabel('mse')
    plt.show()


def get_weight_dict(w0, rev_wd):
    results = {}
    for i in rev_wd:
        results[rev_wd[i]] = w0[i]
    return results


def get_results_vec(results_dict, wd):
    vec = np.zeros(len(wd))
    for idx in results_dict:
        vec[wd[idx]] = results_dict[idx]
    return vec


def run_complex(rf, last_x_chunk, last_y_chunk, reader):
    wd = {}         # tree index to row number dictionary
    rev_wd = {}     # row number to tree index dictionary
    w0 = []
    i = 0
    for idx in rf.get_idx_list():
        wd[idx] = i
        rev_wd[i] = idx
        i += 1
        w0.append(1.0 / n_trees)
    R0 = np.identity(n_trees)
    R0i = np.linalg.inv(R0)
    T = 0.9
    ss = 0.0001
    replace_flag = 1
    normalization_flag = 1
    mse = 0.0
    mse_vec = []

    n_test_data = 200000
    for i in range(chunk_size, chunk_size + n_test_data):
        data_point = reader.read_data_point()
        x = data_point['x']
        y = data_point['y']
        predict = rf.predict_weighted_sum(x, get_weight_dict(w0, rev_wd))

        # print the result
        print 'i = {0}\tpredict = {1:.5f}\ttarget = {2:.5f}\tmse = {3:.5f}'.format(i, predict, y, mse / (i + 1 - chunk_size))
        mse += (predict - y) ** 2
        mse_vec.append(mse / (i + 1 - chunk_size))

        # update the last chunk
        last_x_chunk.append(x)
        last_y_chunk.append(y)
        del last_x_chunk[0]
        del last_y_chunk[0]

        # complex version
        F = get_results_vec(rf.predict_results(x), wd)
        u = F * math.sqrt(1 - T)
        ut = np.transpose(u)
        RF = R0 * T + np.outer(u, ut)
        RFi = np.linalg.pinv(RF)
        wF = w0 + ss * np.dot(RFi, F) * (y - np.dot(np.transpose(w0), F))

        # update value
        w0 = wF
        R0 = RF
        R0i = RFi

        if replace_flag == 1:
            # replace trees
            # delete old trees
            idx_list = []
            threshold = 1.0 / n_trees * 0.9
            for idx in wd:
                if w0[wd[idx]] < threshold:
                    w0[wd[idx]] = 1.0 / n_trees
                    idx_list.append(idx)
                    rev_wd[wd[idx]] = -1
            if len(idx_list) == 0:
                continue
            for idx in idx_list:
                del wd[idx]
            rf.delete(idx_list)
            print 'replace {0} trees whose indices are {1}'.format(len(idx_list), idx_list)
            # insert new trees
            rf.insert_with_random_forest_regressor(len(idx_list), last_x_chunk, last_y_chunk)
            # reassign row numbers to new trees
            for idx in rf.get_idx_list():
                if not (idx in wd.keys()):
                    for j in range(n_trees):
                        if rev_wd[j] == -1:
                            wd[idx] = j
                            rev_wd[j] = idx
                            break
        if normalization_flag == 1:
            # normalization
            w0 = w0 / w0.sum()
    print 'mse = ', mse / n_test_data
    plt.plot(mse_vec)
    plt.ylabel('mse')
    plt.show()


# main function
# open the file
directory = "/Users/dengjingyu/nyu/Research/data/"
data_file_name = "predictors_no_noise.csv"
target_file_name = "response_no_noise.csv"
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
    y.append(data_point['y'])
rf = OnlineEnsemble()
rf.insert_with_random_forest_regressor(n_estimators=n_trees, x=x, y=y)
last_x_chunk = x
last_y_chunk = y

# start test
run_simple(rf, last_x_chunk, last_y_chunk, reader)
#run_complex(rf, last_x_chunk, last_y_chunk, reader)


end = time.time()
print end - start