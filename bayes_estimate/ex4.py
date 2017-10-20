#encoding: utf8

from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import beta 
import numpy as np

'''
Suppose that 50 people are given a placebo and 50 are given a new
treatment. 30 placebo patients show improvement while 40 treated patients
show improvement. Let tau = p2-p1 where p2 is the probability of
improving under treatment and p1 is the probability of improving under
placebo.

(a) Find the mle of tau. Find the standard error and 90 percent confidence
    interval using the delta method.

(b) Find the standard error and 90 percent confidence interval using the
    parametric bootstrap.

(c) Use the prior f(p1, p2) = 1. Use simulation to find the posterior
    mean and posterior 90 percent interval for τ .

(d) Let

        ψ = log((p1/(1-p1) / (p2/(1-p2)))

  be the log-odds ratio. Note that ψ = 0 if p1 = p2. Find the mle of ψ.
  Use the delta method to find a 90 percent confidence interval for ψ.

(e) Use simulation to find the posterior mean and posterior 90 percent
  interval for ψ.

alogrithm:


'''

def ex4():
    p1_hat = 30./50
    p2_hat = 40./50

    tau_hat = p2_hat - p1_hat #why? 同变性. 泛函不变性

    alpha = 0.1
    '''
    (a) confidence interval by delta method
    p_hat = p2_hat - p1_hat
    f(x1,x2;p1,p2) = f(x1;p1)*f(x2;p2) \properto p1**x1*(1-p1)**(n-x1)*p2**x2*(1-p2)**(n-x2)    ;X1,X2 independent
    logf = x1*logp1 + (n-x1)log(1-p1) + x2*logp2 + (n-x2)log(1-p2)
    nabla logf = [x1/p1 - (n-x1)/(1-p1); x2/p2 - (n-x2)/(1-p2)]
    nabla2 logf = [-x1/p1^2 - (n-x1)/(1-p1)^2, 0; 0, -x2/p2^2-(n-x2)/(1-p2)^2]
    I(p) = [-E(nabla2 logf)]
         = [E(X1/p1^2 + (n-X1)/(1-p1)^2) 0; 0, E(X2/p2^2 + (n-X2)/(1-p2)^2)] 
         = [n/p1 + n/(1-p1), 0; 0, n/p2 + n/(1-p2)]
         = [n/[p1(1-p1)], 0; 0, n/[p2(1-p2)] ]

    J(p) = I.inv = [p1(1-p1)/n, 0; 0, (1-p2)p2/n]

    g_deriv = [-1,1] 
    delta method:
    se(tau_hat) = np.sqrt(g'.T * J * g')
    '''
    Jn = np.array([[p1_hat * (1-p1_hat)/50, 0], [0, p2_hat*(1-p2_hat)/50]])
    g_deriv = np.array([-1., 1])
    delta = np.dot(g_deriv.T, Jn)
    delta = np.dot(delta, g_deriv)
    se_tau_hat = np.sqrt(delta)
    dis = se_tau_hat * norm.ppf(1-alpha/2)
    print "tau hat: %.4f" % tau_hat
    print 'confidence interval by delta method: [%.4f, %.4f]' % (tau_hat-dis, tau_hat+dis)
    
    '''
    b. confidence interval by bootstrap
    '''
    B = 1000
    bt_sampls = np.zeros(B)
    p1_samples = binom.rvs(n=50, p=p1_hat, size=B) / 50.
    p2_samples = binom.rvs(n=50, p=p2_hat, size=B) / 50.
    tau_samples = p2_samples - p1_samples
    se_tau_hat = np.sqrt(tau_samples.var())
     
    dis = se_tau_hat * norm.ppf(1-alpha/2)
    print 'confidence interval by bootstrap: [%.4f, %.4f]' % (tau_hat - dis, tau_hat+dis)

    '''
    c.  find the posterior mean and posterior interval for tau
    f(x1,x2|p1,p2)
    = f(x1|p1)f(x2|p2) \properto p1^x1*(1-p1)^(n-x1)*p2^x2*(1-p2)^(n-x2)
    = Beta(x1+1, n-x1+1)*Beta(x2+1, n-x2+1)

    f_postorior \properto Beta(31, 21)*Beta(41,11)
    f_postorior(p1|x1,x2) = \int f(x1,x2|p1,p2) dp2 = Beta(31,21)
    f_postorior(p2|x1,x2) = \int f(x1,x2|p1,p2) dp1 = Beta(41,11)

    f_postorior(p1,p2|x1,x2) = f_postorior(p1|x1)*f_postorior(p2|x1)

    tau_post_hat = p2_post_hat - p1_post_hat
    '''
    p2_post_hat = 41./(41 + 11) 
    p1_post_hat = 31./(31 + 21) 
    tau_post_hat = p2_post_hat - p1_post_hat

    B = 1000
    p1_poster_samples = beta.rvs(a=31,b=21,size=B)
    p2_poster_samples = beta.rvs(a=41,b=11,size=B)

    tau_post_samples = p2_poster_samples - p1_poster_samples
    tau_post_samples.sort()
    ta = tau_post_samples[int(B*alpha/2)] 
    tb = tau_post_samples[int(B*(1-alpha/2))] 
    print "the tau post mean: %.4f" % tau_post_samples.mean()
    print "the tau post interval is: [%.4f, %.4f]" % (ta, tb)
    

    '''
    d. find mle of log-odd radio, and the 90% confidence interval 
        ψ = log(p1/(1-p1) / (p2/(1-p2))) = log(p1) - log(1-p1) - log(p2) + log(1-p2)
        phi_deriv = np.array([1/p1 + 1/(1-p1), -1/p2 - 1/(1-p2)])
    '''
    phi_hat = np.log((p1_hat/(1-p1_hat))/(p2_hat/(1-p2_hat)))
    phi_deriv = np.array([1/p1_hat + 1/(1-p1_hat), -1/p2_hat-1/(1-p2_hat)])

    delta = np.dot(phi_deriv.T, Jn)
    delta = np.dot(delta, phi_deriv)
    se_phi_hat = np.sqrt(delta) 
    dis = se_phi_hat * norm.ppf(1-alpha/2)
     
    print 'mle of log-odd ratio is: %.4f; se of phi: %.4f' % (phi_hat, se_phi_hat)
    print 'confidence interval by delta method: [%.4f, %.4f]' % (phi_hat-dis, phi_hat+dis)

    '''
    (e) Use simulation to find the posterior mean and posterior 90 percent interval for ψ.
    '''
    phi_post_samples = np.log(p1_poster_samples) - np.log(1-p1_poster_samples) \
                      - np.log(p2_poster_samples) + np.log(1-p2_poster_samples)

    phi_post_samples.sort()
    phi_post_mean = phi_post_samples.mean()
    ta = phi_post_samples[int(B*alpha/2)] 
    tb = phi_post_samples[int(B*(1-alpha/2))] 

    print "the phi post mean: %.4f" % phi_post_samples.mean()
    print "the phi post interval is: [%.4f, %.4f]" % (ta, tb)


if __name__ == '__main__':
    ex4()




