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
x = []
y = []
for i in range(chunk_size):
    data_point = reader.read_data_point()
    x.append(data_point['x'])
    y.append(data_point['y'])
rf = OnlineEnsemble()
rf.insert_with_rf(n_estimators=n_trees, x=x, y=y)
last_x_chunk = x
last_y_chunk = y

# start test
w0 = {}
for idx in rf.get_idx_list():
    w0[idx] = 1.0 / n_trees

ss = 0.0001
mse = 0.0

n_test_data = 100
for i in range(chunk_size, chunk_size + n_test_data):
    data_point = reader.read_data_point()
    x = data_point['x']
    y = data_point['y']
    predict = rf.predict_weighted_sum(x, w0)

    # print the result
    print 'i = {0}\tpredict = {1:.5f}\ttarget = {2:.5f}'.format(i, predict, y)
    mse += (predict - y) ** 2

    # update the last chunk
    last_x_chunk.append(x)
    last_y_chunk.append(y)
    del last_x_chunk[0]
    del last_y_chunk[0]

    # simple version
    results = rf.predict_results(x)
    for idx in results:
        w0[idx] += results[idx] * ss * (y - predict)

    """
    # delete or inserting trees
    idx_list = []
    cnt = 0
    threshold = 1.0 / n_trees * 0.8
    for j in range(n_trees):
        if w0[j] < threshold:
            idx_list.append(j)
            cnt += 1
    """
print 'mse = ', mse / n_test_data