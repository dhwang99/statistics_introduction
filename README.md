# statistics_introduction
概率统计学入门。实现一些概率、采样、统计相关的算法

### 1. 概率入门的一些算法
####    1). 不同分布随机数的生成。

>  包括从uniform生成符合正态分布、符合指数分布的随机数; 根据标准正态分布表汇制不同mu,sigma参数的正态分布图; 生成指数分布图；根据数据集合，生成直方图

>       code: [probility/gen_distribute_from_U.py](probility/gen_distribute_from_U.py)
>       image: [probility/images](probility/images)

####    2). 标准正态分布表
>  主要转换了网上的标准正态分布表，并增加了根据x计算概率，和根据概率计算x的功能
       [probility/standard_normal_dist.py](probility/standard_normal_dist.py)

####    3). 均匀分布的伪随机数生成
>       主要是两个算法：Mersenne_Twister, linear congruential. 算法来自网上
        [probility/random_number_has_U.py](probility/random_number_has_U.py)

####    4). 均匀分布随机数测试概率分布
>        p=0.3 时，随试验次数增加，基本稳定在0.3 +- 0.005 左右; p=0.03时，随试验次数增加，偏差还是比较大 0.03 +- 0.005;
        即：概率越小，随试验次数增加，偏差还是比较大
>       code: [probility/test_p.py](probility/test_p.py)
>       image: [probility/images](probility/images)
 
####    5). 一些有意的小程序
          a. 计算乐透中奖的概率.
          b. 计算乐透中奖率 > 0.5时，所需彩票数. 
          c. ...
           [probility/small_tips.py](probility/small_tips.py)    
