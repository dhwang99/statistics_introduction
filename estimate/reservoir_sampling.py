#encoding: utf8
import numpy as np
import sys

'''
Reservoir_Sampling, 蓄水池采样算法。
这种方法比较精巧，用在一些常用的采样过程中，算法实现一下, 用来抽取行样本
'''

def reservoir_sampling(N, filename):
    k = 0
    lines = []
    with open(filename) as fp:
        while True:
            line = fp.readline()
            if not line:
                break

            line = line.strip()
            
            if k < N:
                lines.append(line)
            else:
                r = np.random.randint(k)
                if r < N:
                    #accept No. k sample and replace r sample
                    lines[r] = line

            k = k + 1

    for line in lines:
        print line


if __name__ == "__main__":
    N = int(sys.argv[1])
    reservoir_sampling(N, sys.argv[2])
