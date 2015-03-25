import random

d = 10  # the number of dimensions
kk = 0.8  # the number of changing dimensions
t = 0.1  # the magnitude of the change
N = 1000  # the number of records in a change
nc = 1000  # the number of changes

for tt in range(1, 11, 1):
    t = float(tt) / 10
    for kkk in range(2, 10, 2):
        kk = float(kkk) / 10
        name_prefix = 'd' + str(d) + '_k' + str(kk) + '_t' + str(t)
        print name_prefix

        data_out = open(name_prefix + '.in', 'w')
        target_out = open(name_prefix + '.out', 'w')

        # initialize weights ai
        a = []
        for i in range(d):
            a.append(random.random())

        # initialize directions si
        s = []
        for i in range(d):
            s.append(-1 + random.randint(0, 1) * 2)

        # start
        for i in range(nc):
            for j in range(N):
                x = []
                y = 0.0
                for k in range(d):
                    x.append(random.random())
                    y += x[k] * a[k]
                #print ','.join(map(str, x))
                data_out.write(','.join(map(str, x)) + '\n')
                if y < sum(a) / 2:
                    #print 0
                    target_out.write('0\n')
                else:
                    #print 1
                    target_out.write('1\n')
                # change weights
                for k in range(int(d * kk)):
                    a[k] += s[k] * t / N
            # reverse direction
            if random.random() < 0.1:
                for k in range(int(d * kk)):
                    s[k] = -s[k]

        data_out.close()
        target_out.close()