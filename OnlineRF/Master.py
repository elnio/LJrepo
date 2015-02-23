import numpy as np
import math
from OnlineRF import OnlineRF
from DataReader import DataReader

directory = "/Users/dengjingyu/nyu/Research/data/"
data_file_name = "predictors_no_noise.csv"
target_file_name = "response_no_noise.csv"
reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

n_trees = 100
dim = 50
num = 1000000
chunk_size = 1000

# initialize the random forest
X = np.ndarray(shape=(chunk_size, dim - 1))
y = np.ndarray(shape=chunk_size)
for i in range(chunk_size):
    data_point = reader.read_data_point()
    X[i] = data_point['X']
    y[i] = data_point['y']
rf = OnlineRF(n_estimators=n_trees)
rf.fit(X, y)

# initialize parameters
w0 = np.transpose(np.ones(shape=n_trees) / n_trees)
R0 = np.identity(n_trees)
R0i = np.linalg.inv(R0)

T = 0.5
ss = 0.0001
mse = 0.0


# start
for i in range(chunk_size, chunk_size + 110000):
    data_point = reader.read_data_point()
    X = data_point['X']
    y = data_point['y']

    F = np.transpose(rf.predict(X))
    print np.dot(np.transpose(w0), F)
    print y
    print i
    # if i % 100 == 0:
    #    print w0
    mse += (np.dot(np.transpose(w0), F) - y) ** 2
    """
    # simple version
    wF = w0 + F * ss * (y - np.dot(np.transpose(w0), F))
    w_min = wF.min()
    if w_min < 0:
        for j in range(n_trees):
            wF[j] += w_min
    wF = wF / wF.sum()
    w0 = wF
    """
    """
    # complex version
    u = F * math.sqrt(1 - T)
    ut = np.transpose(u)
    RF = R0 * T + np.dot(u, ut)
    # RFi = np.linalg.inv(RF)
    tmp1 = np.dot(np.dot(R0i, np.dot(u, ut)), R0i)
    tmp2 = np.dot(np.dot(ut, R0i), u)
    RFi = 1 / T * (R0i - tmp1) / (1 / T + tmp2)
    wF = w0 + ss * (y - np.dot(np.transpose(w0), F)) * np.dot(RFi, F)
    wF = wF / wF.sum()
    w0 = wF
    R0 = RF
    R0i = RFi
    """

    """
    #replace trees based on a threshold
    rf.update_chunk(X, y)
    flag = np.zeros(n_trees)
    cnt = 0
    for j in range(n_trees):
        if w0[j] < (1.0/n_trees)/3:
            flag[j] = 1
            cnt += 1
            w0[j] = 1.0/n_trees
    if cnt > 0:
        rf.replace_tree(flag, cnt)
    w0 = w0 / w0.sum()
    """

print mse / 110000

reader.close()