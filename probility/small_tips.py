#encoding: utf8

'''
准备在这儿放一些有意思的小测试程序。
'''

'''
有n张彩票，m张中奖。
问：
1. 抽多少次会有50%的中奖率
2. 抽多少次后一定会中奖

p1 + ((1-p1)*p2 + (1-p2)*p3 ....)
'''

def lotto_probility(n,m, ep):
    count = 0
    p = 0.
    while True:
        pn = m * 1. / (n - count)
        if pn > 1.:
            pn = 1

        p = p + (1. - p ) * pn

        count += 1

        if p >= ep:
            break

    return count

n=20
m =5
ep = 1.
for i in range(1, 11):
    ep = i / 10. 
    c = lotto_probility(n, m, ep)
    print "total: %s, wa: %s, eppect %s, try times: %s" % (n,m,ep, c)

