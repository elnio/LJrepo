import numpy as np
import math
from online_ensemble import OnlineEnsemble
from data_reader import DataReader

# open the file
directory = "/Users/dengjingyu/nyu/Research/data/"
data_file_name = "predictors_no_noise.csv"
target_file_name = "response_no_noise.csv"
reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

# set parameters
n_trees = 100
dim = 50
chunk_size = 1000

# initialize the random forest
x = np.ndarray(shape=(chunk_size, dim - 1))
y = np.ndarray(shape=chunk_size)
for i in range(chunk_size):
    data_point = reader.read_data_point()
    x[i] = data_point['x']
    y[i] = data_point['y']
rf = OnlineEnsemble()
rf.first_build(n_estimators=n_trees, x=x, y=y)
last_x_chunk = x
last_y_chunk = y
last_idx = 0

# start test
w0 = np.transpose(np.ones(shape=n_trees) / n_trees)
R0 = np.identity(n_trees)
R0i = np.linalg.inv(R0)

T = 0.8
ss = 0.0001
mse = 0.0

n_test_data = 100;
for i in range(chunk_size, chunk_size + n_test_data):
    data_point = reader.read_data_point()
    x = data_point['x']
    y = data_point['y']
    predict = rf.predict_weighted_sum(x, w0)

    # print the result
    print 'i = {}, predict = {}, target = {}'.format(i, predict, y)
    mse += (predict - y) ** 2

    # update the last chunk
    last_x_chunk[last_idx] = x
    last_y_chunk[last_idx] = y
    last_idx = (last_idx + 1) % chunk_size

    # simple version
    F = rf.predict_results(x)
    wF = w0 + F * ss * (y - predict)
    w0 = wF

    # complex version
    F = rf.predict_results(x)
    u = F * math.sqrt(1 - T)
    ut = np.transpose(u)
    RF = R0 * T + np.outer(u, ut)
    RFi = np.linalg.pinv(RF)
    wF = w0 + ss * np.dot(RFi, F) * (y - np.dot(np.transpose(w0), F))
    R0 = RF
    R0i = RFi
    w0 = wF

    # delete or inserting trees
    idx_list = []
    cnt = 0
    threshold = 1.0 / n_trees * 0.8
    for j in range(n_trees):
        if w0[j] < threshold:
            idx_list.append(j)
            cnt += 1