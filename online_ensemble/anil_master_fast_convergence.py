import numpy as np
import matplotlib.pyplot as plt
import math
import time
from online_ensemble import OnlineEnsemble
from data_reader import DataReader



def run_simple(rf, last_x_chunk, last_y_chunk, reader, n_trees, chunk_size, category):
    xx = {}    
    w0 = {}
    w1 = {}
    E0 = {}
    M = {}
    woods = {}
    for idx in rf.get_idx_list():
        w0[idx] = 1.0 / n_trees
        E0[idx] = 0.0
        M[idx] = 1
    w0["intercept"] = 0.0       
    E0["intercept"] = 0.0
    M["intercept"] = 0
    woods[0] = 1
    woods[1] = n_trees      

    # Set the parameters
    ws = 6 # Window size to check the MSE in this window increases or decreases     
    wsr = 0.0003 # Stopping rule for the stochastic gradient descent
    SS = 0.001 # Scaled step-size ( 0.001 works nice! )
    lambda_E0 = 0.01 # Forgetting factor for the cross correlation
    
    # Dull parameters (no need to change)
    reg1 = 0  
    reg2 = 1
    lamm = 0.0000
    
    # Initialize the adaptive parameters
    add_trees = 0 # Number of extra trees . 
    N_trees = n_trees # Size of the forest
    ff1 = np.zeros(ws)
    mse = 0.0
    mse_vec = [0]
    mse_raw = [0]
    rep_vec = [0]
    replace_flag = 0 #2
    f1 = False # Flag to check the convergence of sgd
    f2 = False # Flag to check if MSE increases
    f11 = -40
    ss = SS/N_trees
    #n_test_data = file_length-chunk_size
    n_test_data = 100
    for i in range(chunk_size, chunk_size + n_test_data):
  
        data_point = reader.read_data_point()
        x = data_point['x']
        y = data_point['y']
        predict = rf.predict_weighted_sum(x, w0)
        if abs(predict)>5:
            predict = np.sign(predict)*5
      
        # print the result
        print 'i = {0}\tpre = {1:.5f}\ttar = {2:.5f}\tmse = {3:.5f}\tss = {4:.5f}\tbias = {5:.3f}\tM ={6}\tsw ={7}\twoods={8}'.format(i, predict, y, mse / (i + 1 - chunk_size), ss, w0['intercept'], N_trees, sum(w0.values()), len(w1))        
        mse += (predict - y) ** 2
        mse_raw.append((predict - y) ** 2) # Keep track of instantenous MSE
        mse_vec.append(mse / (i + 1 - chunk_size)) 
        
        # update the last chunk
        last_x_chunk.append(x)
        last_y_chunk.append(y)
        del last_x_chunk[0]
        del last_y_chunk[0]

        # simple version
        results = rf.predict_results(x)
        results["intercept"] = 1.0
        for idx in M:
            w1[M[idx]] = 0
            xx[M[idx]] = 0
        for idx in results:
            xx[M[idx]] += results[idx]/woods[M[idx]]
        ss = float(SS / (np.linalg.norm( xx.values())**2))
        for idx in results:
            if idx != "intercept":     
                w0[idx] *= 1-lamm
                w1[M[idx]] += w0[idx]
            else:                                                      
                w1[idx] = w0[idx]
            dw0 = results[idx] * ss * (y - predict)
            w1[M[idx]] += dw0/woods[M[idx]]                  # Update the weights
            E0[idx] *= 1-lambda_E0          # Update the Correlation vector
            E0[idx] += results[idx] * y * lambda_E0
        for idx in results:
            w0[idx] = w1[M[idx]]/woods[M[idx]]

        # Check the convergence
        for w in range(ws-1):
            ff1[w] = ff1[w+1]
        ff1[ws-1] = np.linalg.norm(xx.values()) * ss * (y - predict)
        f11 += 1
        f1 = (np.mean(ff1[0:ws-1]) < np.linalg.norm(w1.values())*wsr)
              
        # Check the error behaviour
        f2 = (np.mean(mse_raw[len(mse_raw)-ws-1:len(mse_raw)-1]) > np.mean(mse_raw[len(mse_raw)-ws*3/2:len(mse_raw)-ws/2]))
        if f2:
            if (add_trees > -math.ceil(N_trees/20)) and ( N_trees + add_trees > n_trees/4):
                add_trees += -math.ceil(N_trees/100)

            
        # replace trees
        n_rep = 0
        if ((f1) or (add_trees < -N_trees/20) )and (f11>0):
            f11 = - len(w1.values())*5 #floor(N_trees/5)
            if (f1) and (~f2) and ( N_trees + add_trees < n_trees*2):
                add_trees += math.ceil(N_trees/10)

  
            # delete old trees
            idx_list = []
            cros = [(reg1 + reg2*w0.values()[j])*E0.values()[j] for j in range(len(w0.values()))]
            cros = cros[0:len(cros)-1]          
            
            # Choose the threshold such that, small values of E0*w0 will be replaced while guaranteeing at most 30 percent of the trees will be replaced
            threshold = min([(np.percentile(cros, 50))*0.5, (np.percentile(cros, 50))])
            for idx in w0:
                if (((reg1 + reg2*w0[idx])*E0[idx]) < threshold) and (idx != "intercept"):
                    idx_list.append(idx)                    
            m = len(idx_list) + 1
            
            # The amount of increase in the size of the forest 
            if add_trees < 0:
                add_trees = max(add_trees, -m+1)
                
            if m == 1:
                continue
                    
            for idx in idx_list:
                del w0[idx]
                del E0[idx]
                if woods[M[idx]] > 1:
                    woods[M[idx]] += -1
                else:
                    del woods[M[idx]]
                    del w1[M[idx]]
                    del xx[M[idx]]
                del M[idx] ###                
            rf.delete(idx_list)
            print 'replace {0} trees whose indices are {1}'.format(len(idx_list), idx_list)
            
            # Update the variables
            n_rep = len(idx_list) + add_trees
            N_trees += add_trees # New size of the forest            
            add_trees = 0
            woods[i] = 0            
            # insert new trees
            rf.insert(len(idx_list)+add_trees, last_x_chunk, last_y_chunk, category=category)
            for idx in rf.get_idx_list():
                if not (idx in w0.keys()):
                    w0[idx] = 1.0 / N_trees
                    E0[idx] = w0[idx]*y
                    M[idx] = i
                    woods[i] += 1
            if woods[i] == 0:
                del woods[i]
        rep_vec.append(n_rep)
       
    print 'mse = ', mse / n_test_data
    return mse_vec, rep_vec, mse

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(mse_vec, 'r')
    ax1.set_ylabel('mse')
    ax2 = ax1.twinx()
    ax2.plot(rep_vec, 'b')
    ax2.set_ylabel('num of replacements')
    plt.show()
    """


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


def test(n_jobs, n_trees, category, chunk_size):
    # main function
    # open the file
    directory = "./data/"
    data_file_name = "challengePredictors.csv"
    target_file_name = "challengeResponse.csv"
    #data_file_name = "predictors_no_noise.csv"
    #target_file_name = "response_no_noise.csv"


    # set parameters
    #n_trees = 1000
    dim = 50
    #chunk_size = 200
    file_length = 10000
    #n_jobs = 1
    #category = 'random_forest_regressor'

    reader = DataReader(num=file_length, data_path=directory + data_file_name, target_path=directory + target_file_name)

    # initialize the random forest
    start = time.time()

    x = []
    y = []
    for i in range(chunk_size):
        data_point = reader.read_data_point()
        x.append(data_point['x'])
        y.append(data_point['y'])
    rf = OnlineEnsemble(n_jobs=n_jobs)
    rf.insert(n_estimators=n_trees, x=x, y=y, category=category)
    last_x_chunk = x
    last_y_chunk = y

    # start test
    mse_vec, rep_vec, mse = run_simple(rf, last_x_chunk, last_y_chunk, reader, n_trees, chunk_size, category)

    end = time.time()
    reader.close()
    exe_time = end - start
    return mse_vec, rep_vec, mse, exe_time