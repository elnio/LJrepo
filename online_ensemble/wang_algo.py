from data_reader import DataReader
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

directory = "/Users/dengjingyu/nyu/Research/data/"
data_file_name = "predictors1.csv"
target_file_name = "classes1.csv"
reader = DataReader(num=1000000, data_path=directory + data_file_name, target_path=directory + target_file_name)

chunk_size = 5000
n_test_chunks = 100
n_classifiers = 10  # number of random forests
test_size = 0.2     # test_size in cross validation
min_samples_leaf = 1

ensemble = []
weights = []
errors = 0.0


for i in range(n_test_chunks):
    # read data chunk
    x, y = reader.read_data_chunk(chunk_size)

    # test previous classifiers
    for k in range(chunk_size):
        xk = x[k]
        yk = y[k]
        vote_dict = {}
        max_vote = 0.0
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
    print 'tested {0} chunks {1} data points, error_rate = {2:.5f}'.format(i + 1,
                                                                           (i + 1) * chunk_size,
                                                                           errors / ((i + 1) * chunk_size))

    # train a new classifier
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=test_size)
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    dt.fit(x_train, y_train)
    ensemble.append(dt)
    print dt.score(x_test, y_test)
    break
    MSE = 0.0
    proba_list = dt.predict_proba(x_test)
    for k in range(len(x_test)):
        xk = x_test[k]
        yk = y_test[k]
        proba = proba_list[k][int(yk)]
        print 'proba', proba
        MSE += (1 - proba) ** 2
    MSE /= len(x_test)
    print 'MSE', MSE
    weights.append(0.25 - MSE)


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
        weights[j] = 0.25 - MSEj

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
        if best < 0:
            break
        new_ensemble.append(ensemble[best])
        new_weights.append(weights[best])
        del ensemble[best]
        del weights[best]
    ensemble = new_ensemble
    weights = new_weights
    print weights
reader.close()