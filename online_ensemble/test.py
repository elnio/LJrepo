import master_classification as my
import wang_algo as wang
import single as single
import matplotlib.pyplot as plt

d = 10  # the number of dimensions
kk = 0.2  # the number of changing dimensions
t = 0.1  # the magnitude of the change
n_test_chunks = 900
name_prefix = 'd' + str(d) + '_k' + str(kk) + '_t' + str(t)
data_file_name = name_prefix + '.in'
target_file_name = name_prefix + '.out'
plt.plot(my.run_classification(data_file_name, target_file_name, n_test_chunks))
plt.plot(wang.run_classification(data_file_name, target_file_name, n_test_chunks))
plt.plot(single.run_classification(data_file_name, target_file_name, n_test_chunks))
plt.legend(['our', 'wang', 'single'], loc='upper right')
plt.show()