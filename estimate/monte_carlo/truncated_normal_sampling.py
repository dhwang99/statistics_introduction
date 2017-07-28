#encoding: utf8

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

'''
truncated normal distribution:

f(x;mu,sigma,a,b) = 1/sigma *phi((x-mu)/sigma)/(Phi((b-mu)/sigma) - Phi((a-mu)/sigma))

'''

def get_truncated_normal_base(TN_param):
    mu, sigma, a, b = TN_param
    d = norm.cdf((b-mu)/sigma) - norm.cdf((a-mu)/sigma) 
    return d

'''
gen truncated normal samples by acceptance_rejection method. Q = U
'''
def truncated_normal_samples_by_AR(n, TN_param):
    mu, sigma, a, b = TN_param
    base = get_truncated_normal_base(TN_param)
    k = 1/base * norm.pdf(0) / sigma * (b-a)  #qs = kU 

    q_samples = a + np.random.random(n) * (b-a)
    acc_samples = []

    for x in q_samples: 
        pdf = 1/base * norm.pdf((x-mu)/sigma)/sigma

        u = np.random.random(1)
        if u <= pdf/k:
            acc_samples.append(x)
    
    return acc_samples

if __name__ == "__main__":
    n = 100000
    TN_param = np.array([1., 1., 0., 4])
    TN_param = np.array([1., 2., 0., 4])
    mu, sigma, a, b = TN_param
    base = get_truncated_normal_base(TN_param)

    TN_acc_samples = truncated_normal_samples_by_AR(n, TN_param)
    
    sample_count = len(TN_acc_samples)
    group_count = int(np.sqrt(sample_count) + 0.999) / 5
    bins = np.linspace(a, b, group_count) 
    hist = np.histogram(TN_acc_samples, bins, normed=True)[0]
  
    plt.plot(bins[:-1]+(bins[1]-bins[0])/2, hist, color='k')  #取中间点
    
    bins = np.linspace(a, b, 1000) 
    xs = map(lambda x:norm.pdf((x-mu)/sigma)/(base*sigma), bins)
    plt.plot(bins, xs, color='r')

    plt.savefig('images/truncnorm_accept_sigma_%s.png' % sigma, format='png')
