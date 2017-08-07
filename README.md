# statistics_introduction
概率统计学入门。实现一些概率、采样、统计相关的算法

## 1. 概率入门的一些算法

####    1). 均匀分布的伪随机数生成
>   主要是两个算法：Mersenne_Twister(梅森旋转法), linear congruential(线性同余法). 算法来自网上.
    里面有测试例子。从例子可以看出，相同的种子序列一样(这也是伪随机数)

>   [probility/gen_random_number.py](probility/gen_random_number.py)

####    2). 标准正态分布表
>  主要转换了网上的标准正态分布表，并增加了算标准的F<sub>x</sub>, F<sub>X</sub><sup>-1</sup>的功能

>   [probility/standard_normal_dist.py](probility/standard_normal_dist.py)


####    3). 不同分布随机数的生成

>  包括从uniform生成符合正态分布、符合指数分布的随机数; 根据标准正态分布表汇制不同mu,sigma参数的正态分布图; 

>   用pdf生成正态、指数分布图；根据采样数据集合，生成直方图. 采用的是随机变量积分变换的方法生成不同的分布

>   备注下：用plt.bar生成柱状图时，需要 加条件:align='edge', 默认取 middle

>   code:[probility/transformations_of_random_variables.py](probility/gen_distribute_from_U.py)

>   images:[probility/images](probility/images)

####    4). 均匀分布随机数的频率分布

>   p=0.3 时，随试验次数增加，基本稳定在0.3 +- 0.005 左右; p=0.03时，随试验次数增加，偏差还是比较大 0.03 +- 0.005;

>    即：概率越小，偏差越大；但随试验次数增加，偏差在减小。如(1000, 10000, 100000) (random_dist003.png, random_dist002,random_dist001)

>   code: [probility/test_p.py](probility/test_p.py)

>   image: [probility/images](probility/images)

####    5). 随机游走
>   从例子看，走n步后，观测值与期望相差变化还是比较明显。

>   试验了5次，分别取p=0.1, 0.2, 0.3, 0.4, 0.5, 绘制了全部步数、最后100步中 期望值、观测值、方差。从图上直观的得到上述结论

>   code:[probility/random_walk.py](probility/random_walk.py)

>   image:[probility/images/random_walk](probility/images/random_walk)

####    6). 样本分布
>   Y=sum(Xi). Xi比较大的情况下，Xi的均值、S2是X的期望和方差的无偏估计(和真实值的期望相等), 且估计的方差为 V(X)/n

>   代码中以X~U(0,1)模拟生成Xn的样本, 然后求Xn的均值、方差，并绘制观测值与真实分布的期望/方差的对比图

>   code:[probility/sample_distribution.py](probility/sample_distribution.py)

>   image:[probility/images/sample_distribution](probility/images/sample_distribution)

####    7). 概率不等式
>   测试了两个概率不等式： markov, chebyshev。 其它的参考笔记吧

>   [probility/test_inequality.py](probility/test_inequality.py)

####    8). Poisson vs Binomial分布对比
>   取 p: 0.05 ~ 0.95, n:10 ~ 100, m: 0 ~9, 来测试possion分布和Binomial分布计算的误差。结果看出来，当p比较少时，两者差别不明显

>   此外，从结果也可以看出，分布还是围绕 期望分布, 在期望附近，pmf高于其它

>  [probility/test_poisson_VS_binomial.py](probility/test_poisson_VS_binomial.py)


####  9) 概率收敛性测试
>   主要写了两个收敛性的测试：大数定律、中心极限定理。Xi样本分别来自均匀分布、指数分布、二项分布。代码和图见下

>   code:[probility/test_convergence.py](probility/test_convergence.py)

>   image:[probility/images/convergence](probility/images/convergence)
 
####    x). 一些有意思的小程序
>   a. 计算乐透中奖的概率.

>   b. 计算乐透中奖率 > 0.5时，所需彩票数. (不放回抽样)

>   c. 计算 sim(x) = (1+1/x)^x 与 np.e的偏差。从试验结果看，x = 50时，相关就在1%以内了. 这也意味着，对possion分布而言，
       (1 - p)^n 

>   d. ...

>   [probility/small_tips.py](probility/small_tips.py)

## 2. 统计推断入门的一些算法

####  1). maximum_likelihood
>    用最大似然估计指数、正态、贝努利分布的参数（正态估计了位置、形状参数）
>    极大似然估计是求参数使得l取极大值，也可以用数值方法求解：如牛顿法、梯度法、A_star算法等等

>    code:[estimate/maximum_likelihood.py](estimate/maximum_likelihood.py)

####  2). EM求解GMM问题 
>   用EM算法估计学生身高分布中，学生比例（多项式分布）参数、身高分布（正态分布）参数。
>   从试验中可以看出，如果两个正态分布的mu差别较大，估计出来的值会准一些；此外，样本适当多点

>    code:[estimate/EM_for_GMM.py](estimate/EM_for_GMM.py)
>   image:[estimate/images/mix_norm](estimate/images/mix_norm)

####  3). monte carlo方法
>    a) 积分模拟计算. 用求期望的方式求积分。E(g(X)) = sum(g(X)f(X)), f(X)为均匀分布; 或者为原函数的分布, g(X)根据样本值取0,1
>       从试验结果看，采样数要比较大, 结果和真值更接近。比如 >= 100万
>       增加了pi的简单模拟计算.
>    code: [estimate/monte_carlo/intify_sim.py](estimate/monte_carlo/intify_sim.py)

>    b) acceptance-reject sampling.接受、拒绝采样：即以 Mq(x)/p(x)的概率接受q(x)产生的样本. 生成了三角分布的样本, q=U
>    code: [estimate/monte_carlo/acceptance_rejection_sampling.py](estimate/monte_carlo/acceptance_rejection_sampling.py)
>    image: [estimate/monte_carlo/images](estimate/monte_carlo/images)

>    c) MCMC方法。这时有比较多的示例，包括M-H采样，Gibbs采样，不一一列举了。 
>    有一个问题看起来还没有完全弄明白： M-H采样后，如果y被拒绝，样本值xi是否作为这一步的样本保留下来？
>    <b>从结果看，保留下来，分布更准;  反之，如果不保留，分布飘得很厉害。只是这样样本集合里，重复的样本比较多。</b>
>    code: [estimate/monte_carlo/](estimate/monte_carlo)
>    image: [estimate/monte_carlo/images](estimate/monte_carlo/images)

