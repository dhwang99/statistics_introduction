#encoding: utf8

import numpy as np

'''
lim(1-1/x)^x = 1/pi
'''
def test_resampling(N):
    hit_pos = np.zeros(N)
    samples = np.random.randint(low=1, high=N, size=100)
    
    for n in samples:
        hit_pos[n] += 1
    
    no_hit = 0
    for i in hit_pos:
        if i == 0:
            no_hit += 1
    
    E_no_hit = (1 - 1./N)**N
    no_hit /= (1. * N)
    print "N: %s; No Hit: %s; expect: %.3f; real ratio: %.3f" % (N, no_hit, E_no_hit, no_hit)
    print "no_hit minux pi:", (no_hit - 1/np.pi)

    return no_hit
    

if __name__ == "__main__":
    N = 100
    sm = 10
    rst = np.zeros(sm)

    for i in xrange(sm):
        rst[i] = test_resampling(N)

    print "Result. mean: %s; std error: %s" % (rst.mean(), rst.std()) 
