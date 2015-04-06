import matplotlib
matplotlib.use('Agg')
import master_classification as my
import wang_algo as wang
import single as single
import matplotlib.pyplot as plt
import time
import datetime
d = 10  # the number of dimensions
kk = 0.2  # the number of changing dimensions
t = 0.1  # the magnitude of the change
for kk in [0.2, 0.4, 0.6]:
    chunk_size = 1000   # the size of chunk
    n_classifiers = 8   # the number of classifiers
    n_test_chunks = 900
    name_prefix = 'd' + str(d) + '_k' + str(kk) + '_t' + str(t)
    data_file_name = name_prefix + '.in'
    target_file_name = name_prefix + '.out'
    outFile = open('results/' + name_prefix + '.txt', 'w')

    for chunk_size in [1000]:
        for n_classifiers in [2, 8, 16]:
            n_test_chunks = 900 * 1000 / chunk_size
            print '------------------', chunk_size, n_classifiers

            plt.clf()

            start_time = time.time()
            error_rate, error_rate_vec = my.run_classification(data_file_name, target_file_name, n_test_chunks, chunk_size, n_classifiers, 'random_forest_regressor')
            end_time = time.time()
            delta = str(int(end_time - start_time))
            plt.plot(error_rate_vec)
            outFile.write(str(chunk_size) + ' ' + str(n_classifiers) + ' ' + str(error_rate) + ' O ' + delta + ' \n')

            start_time = time.time()
            error_rate, error_rate_vec = wang.run_classification(data_file_name, target_file_name, n_test_chunks, chunk_size, n_classifiers)
            end_time = time.time()
            delta = str(int(end_time - start_time))
            plt.plot(error_rate_vec)
            outFile.write(str(chunk_size) + ' ' + str(n_classifiers) + ' ' + str(error_rate) + ' W ' + delta + ' \n')
            """
            start_time = time.time()
            error_rate, error_rate_vec = single.run_classification(data_file_name, target_file_name, n_test_chunks, chunk_size, n_classifiers)
            end_time = time.time()
            delta = str(int(end_time - start_time))
            plt.plot(error_rate_vec)
            outFile.write(str(chunk_size) + ' ' + str(n_classifiers) + ' ' + str(error_rate) + ' S ' + delta + ' \n')
            """
            start_time = time.time()
            error_rate, error_rate_vec = my.run_classification(data_file_name, target_file_name, n_test_chunks, chunk_size, n_classifiers, 'combination')
            end_time = time.time()
            delta = str(int(end_time - start_time))
            plt.plot(error_rate_vec)
            outFile.write(str(chunk_size) + ' ' + str(n_classifiers) + ' ' + str(error_rate) + ' C ' + delta + ' \n')

            plt.legend(['svm', 'wang', 'combination'], loc='upper right')
            plt.title(name_prefix)
            plt.show()
            plt.savefig('results/' + name_prefix + '/ch' + str(chunk_size) + '_n' + str(n_classifiers) + '.png')

    outFile.close()