#encoding: utf8

import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import pdb


'''
X1, X2..., Xn ~ N(mu, 1)
a. simulate a data set (using mu=5) consisting 100 observations

b. take f(mu)=1 and find the posterior desity. plot it

c. simulate 1000 draws from the posterior. Plot a histogram of the simulated values and 
    compare the histogram to the answer in (b).

d. let theta = exp(mu). find the posterior desity for theta analytically and by simulation

e. Find a 95 percent posterior interval for mu.

f. Find a 95 percent confidence interval for theta.
'''

'''
f(mu|xn) 
= prod{1/sqrt(2*pi)exp[-(xi-mu)^2)/2]}*f(mu) 
\propto exp[-sum(xi^2)/2]*exp[-n(x_bar^2-2*mu*x_bar+mu^2)/2]*exp[n*x_bar^2/2]
\proto 1/[sqrt(2*pi)*sqrt(1/n)]*exp{-[x_bar - mu]^2/2[sqrt(1/n)^2]}
\sim N(x_bar, 1/n)

P(theta < t) = P(exp(mu) < t) = P(mu < log(t)) 

g(theta) = 1/sqrt(2pi*1/n) * exp[-(log(theta) - x_bar)^2/2[sqrt(1/n)^2] * 1/theta
'''

def test_ex2():
    mu = 5
    sigma = 1
    n=100
    #step a
    X_samples = norm.rvs(loc=mu, scale=1, size=n)

    X_bar = X_samples.mean()
    posterior_mu = X_bar    #mu的后验分布的mu
    posterior_sigma = np.sqrt(1./n)  #mu的后验分布的标准差 
    
    #step b. plot posterior desity for mu
    x_samples = np.linspace(posterior_mu - 2, posterior_mu+2, 1000)
    x_pdf = norm.pdf(x_samples, loc=posterior_mu, scale=posterior_sigma)
    ls = []
    l = plt.plot(x_samples, x_pdf, color='r', alpha=0.5)
    ls.append(l)

    #step c. plot histogram
    post_samples = norm.rvs(loc=posterior_mu, scale=posterior_sigma, size=1000)
    group_count = int(np.sqrt(1000))
    plt.hist(post_samples, bins = group_count, normed=True, color='b', alpha=0.5)
    plt.legend(handles=ls, labels=['desity for mu'], loc='best')
    plt.savefig('images/post_pdf.png', format='png')
    plt.clf()

    ls = []
    #theta = exp(mu)
    #step d. posterior desity for theta analytically
    t_pos = np.linspace(np.exp(posterior_mu - 2), np.exp(posterior_mu+2), 1000)
    t_pdfs = np.zeros(1000)
    for i in xrange(len(t_pos)):
        t = t_pos[i]
        g = 1/np.sqrt(2*np.pi*1./n*t) * np.exp(-(np.log(t)-X_bar)**2/(2*(np.sqrt(1./n)**2))) / t
        t_pdfs[i] = g

    l = plt.plot(t_pos, t_pdfs, color='r', alpha=0.5)
    ls.append(l)

    #posterior desity for theta by simulation 
    t_samples = np.exp(post_samples)
    plt.hist(t_samples, bins = group_count, normed=True, color='b', alpha=0.5)

    plt.legend(handles=ls, labels=['analytically'], loc='best')
    plt.savefig('images/exp_post_pdf.png', format='png')
    plt.clf()

    #step e:
    alpha = 0.05
    half_dis = posterior_sigma * norm.ppf(1.-alpha/2)
    print "one 95 percent posterior interval for mu: [%.4f, %.4f]" % (posterior_mu - half_dis, posterior_mu + half_dis)

    #step f:
    t_samples.sort()
    beg = int(alpha/2 * 1000)
    end = int((1-alpha/2) * 1000)

    print "one 95 percent confidence interval for theta: [%.4f, %.4f]" % (t_samples[beg], t_samples[end])

if __name__ == '__main__':
    test_ex2()
