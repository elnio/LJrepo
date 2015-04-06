from data_reader import DataReader
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def run_classification(data_file_name="d10_k0.8_t0.1.in", target_file_name="d10_k0.8_t0.1.out", n_test_chunks=100, chunk_size=1000, n_classifiers=8):
    directory = "data/"
    reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

    test_size = 0.4     # test_size in cross validation
    MSEr = 0.25

    ensemble = []
    weights = []
    errors = 0.0
    cnt = 0
    nc = 0
    error_rate = 0.0
    error_rate_vec = [0.0]


    for i in range(n_test_chunks):
        # read data chunk
        x, y = reader.read_data_chunk(chunk_size)

        # test previous classifiers
        for k in range(chunk_size):
            xk = x[k]
            yk = y[k]
            vote_dict = {}
            max_vote = -1000
            if len(ensemble) == 0:
                continue
            for j in range(len(ensemble)):
                predict = int(ensemble[j].predict(xk))
                if not(predict in vote_dict.keys()):
                    vote_dict[predict] = 0.0
                vote_dict[predict] += weights[j]
                if vote_dict[predict] > max_vote:
                    max_vote = vote_dict[predict]
                    res = predict
            if res != yk:
                errors += 1
            cnt += 1
        if len(ensemble) != 0:
            nc += 1
            error_rate = errors / cnt
            error_rate_vec.append(error_rate)
            print 'tested {0} chunks {1} data points, error_rate = {2:.5f}'.format(nc,
                                                                                   nc * chunk_size,
                                                                                   errors / cnt)

        # train a new classifier
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=test_size)
        dt = DecisionTreeClassifier(min_samples_leaf=3)
        dt.fit(x_train, y_train)
        ensemble.append(dt)

        MSE = 0.0
        proba_list = dt.predict_proba(x_test)
        for k in range(len(x_test)):
            xk = x_test[k]
            yk = y_test[k]
            proba = proba_list[k][int(yk)]
            MSE += (1 - proba) ** 2
        MSE /= len(x_test)
        #print 'MSE of new trained classifier:', MSE
        weights.append(MSEr - MSE)


        # update weights of previous classifiers
        for j in range(len(ensemble) - 1):
            MSEj = 0.0
            Cj = ensemble[j]
            proba_list = Cj.predict_proba(x)
            for k in range(chunk_size):
                xk = x[k]
                yk = y[k]
                proba = proba_list[k][int(yk)]
                MSEj += (1 - proba) ** 2
            MSEj /= chunk_size
            weights[j] = MSEr - MSEj
        #print 'updated weights: ', weights

        # keep top classifiers
        new_ensemble = []
        new_weights = []
        while (len(new_ensemble) < n_classifiers) and (len(ensemble) > 0):
            max_weight = 0.0
            best = -1
            for j in range(len(ensemble)):
                if weights[j] > max_weight:
                    max_weight = weights[j]
                    best = j
            if best < 0 and len(new_ensemble) != 0:
                break
            new_ensemble.append(ensemble[best])
            new_weights.append(weights[best])
            del ensemble[best]
            del weights[best]
        ensemble = new_ensemble
        weights = new_weights
        #print weights
    """
    print 'error_rate = ', error_rate
    plt.plot(error_rate_vec)
    plt.ylabel('error_rate')
    plt.show()
    """
    reader.close()
    return error_rate, error_rate_vec