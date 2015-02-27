import numpy as np
import random
from sklearn.ensemble.forest import RandomForestRegressor
f = open("/Users/dengjingyu/Downloads/housing.data.txt")
m = [map(float, line.split()) for line in f]
for row in m:
    del row[3]
avg = 0.0
for i in range(100):
    rl = []
    for j in range(51):
        rd = random.randint(0, 505)
        while rd in rl:
            rd = random.randint(0, 505)
        rl.append(rd)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for j in range(506):
        if j in rl:
            x2.append(m[j])
            y2.append(m[j][12])
        else:
            x1.append(m[j])
            y1.append(m[j][12])
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    x1 = x1[:, :-1]
    x2 = x2[:, :-1]
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x1, y1)
    mse = 0.0
    for j in range(51):
        yy = rf.predict(x2[j])
        mse += (yy - y2[j]) ** 2
    mse /= 51
    print mse
    avg += mse
print avg / 100
f.close()