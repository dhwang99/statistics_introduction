# statistics_introduction
概率统计学入门。实现一些概率、采样、统计相关的算法

### 1. 概率入门的一些算法

####    1). 均匀分布的伪随机数生成
>   主要是两个算法：Mersenne_Twister(梅森旋转法), linear congruential(线性同余法). 算法来自网上.
    里面有测试例子。从例子可以看出，相同的种子序列一样(这也是伪随机数)

####    2). 不同分布随机数的生成

>  包括从uniform生成符合正态分布、符合指数分布的随机数; 根据标准正态分布表汇制不同mu,sigma参数的正态分布图; 
    用pdf生成正态、指数分布图；根据采样数据集合，生成直方图
   备注下：用plt.bar生成柱状图时，需要 加条件:align='edge', 默认取 middle

>   code:[probility/gen_distribute_from_U.py](probility/gen_distribute_from_U.py)

>   images:[probility/images](probility/images)

####    3). 标准正态分布表
>  主要转换了网上的标准正态分布表，并增加了算标准的F<sub>x</sub>, F<sub>X</sub><sup>-1</sup>的功能

>   [probility/standard_normal_dist.py](probility/standard_normal_dist.py)

>   [probility/gen_random_number.py](probility/gen_random_number.py)

####    4). 均匀分布随机数测试概率分布
>   p=0.3 时，随试验次数增加，基本稳定在0.3 +- 0.005 左右; p=0.03时，随试验次数增加，偏差还是比较大 0.03 +- 0.005;
    即：概率越小，偏差越大；但随试验次数增加，偏差在减小。如(1000, 10000, 100000) (random_dist003.png, random_dist002,random_dist001)

>   code: [probility/test_p.py](probility/test_p.py)

>   image: [probility/images](probility/images)
 
####    5). 一些有意思的小程序
>   a. 计算乐透中奖的概率.

>   b. 计算乐透中奖率 > 0.5时，所需彩票数. (不放回抽样)

>   c. ...

>   [probility/small_tips.py](probility/small_tips.py)
