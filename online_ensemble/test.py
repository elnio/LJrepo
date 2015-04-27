import matplotlib
matplotlib.use('Agg')
import anil_master_fast_convergence as my
import wang_algo as wang
import single as single
import matplotlib.pyplot as plt
import time
import datetime

outFile = open('result/result.txt', 'w')
for n_jobs in [1]:
    for chunk_size in [200]:
        for n_trees in [200]:
            name_prefix = str(n_jobs) + '_' + str(chunk_size) + '_' + str(n_trees)
            outFile2 = open('result/' + name_prefix + '.txt', 'w')
            print 'n_jobs=', n_jobs, 'chunk_size=', chunk_size, 'n_trees=', n_trees

            plt.clf()

            mse_vec, rep_vec, mse, exe_time = my.test(n_jobs=n_jobs, chunk_size=chunk_size, n_trees=n_trees, category='random_forest_regressor')
            plt.plot(mse_vec)
            outFile.write(name_prefix + '_' + 'RF_' + str(exe_time) + ' \n')
            outFile2.write(name_prefix + '_' + 'RF_' + str(exe_time) + ' \n')

            mse_vec, rep_vec, mse, exe_time = my.test(n_jobs=n_jobs, chunk_size=chunk_size, n_trees=n_trees, category='svm_regressor')
            plt.plot(mse_vec)
            outFile.write(name_prefix + '_' + 'SVM_' + str(exe_time) + ' \n')
            outFile2.write(name_prefix + '_' + 'SVM_' + str(exe_time) + ' \n')

            mse_vec, rep_vec, mse, exe_time = my.test(n_jobs=n_jobs, chunk_size=chunk_size, n_trees=n_trees, category='combination')
            plt.plot(mse_vec)
            outFile.write(name_prefix + '_' + 'COM_' + str(exe_time) + ' \n')
            outFile2.write(name_prefix + '_' + 'COM_' + str(exe_time) + ' \n')

            outFile2.close()

            plt.legend(['RF', 'SVM', 'COMB'], loc='upper right')
            plt.title(name_prefix)
            plt.show()
            plt.savefig('result/' + name_prefix + '.png')
outFile.close()