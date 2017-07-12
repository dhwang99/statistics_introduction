# encoding: utf8

'''
生成符合均匀分布的伪随机数

梅森旋转
intrudction: https://en.wikipedia.org/wiki/Mersenne_Twister

线性同余法
another pseudo method is : linear congruential generator, is too simple to use. 
  U(0., 1.):
     seed = 1103515245UL * seed + 12345UL
     rand_real_number = double(seed) / -1UL


更复杂的算法: box-mueller, ...

'''

def _int32(x):
    # Get the 32 least significant bits.
    return int(0xFFFFFFFF & x)
'''
梅森旋转. 来自网上
'''
class MT19937:
    def __init__(self, seed):
        # Initialize the index to 0
        self.index = 624
        self.mt = [0] * 624
        self.mt[0] = seed  # Initialize the initial state to the seed
        for i in range(1, 624):
            self.mt[i] = _int32(
                1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)

    def extract_number(self):
        if self.index >= 624:
            self.twist()

        y = self.mt[self.index]

        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18

        self.index = self.index + 1

        return _int32(y)

    def twist(self):
        for i in range(624):
            # Get the most significant bit and add it to the less significant
            # bits of the next number
            y = _int32((self.mt[i] & 0x80000000) +
                       (self.mt[(i + 1) % 624] & 0x7fffffff))
            self.mt[i] = self.mt[(i + 397) % 624] ^ y >> 1

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908b0df
        self.index = 0
        #test

'''
线性同余法. 自己瞎bb了一个
linear congruential generator

因为通过线性同余方法构建的伪随机数生成器的内部状态可以轻易地由其输出演算得知，所以此种伪随机数生成器属于统计学伪随机数生成器。

设计密码学的应用必须至少使用密码学安全伪随机数生成器，故需要避免由线性同余方法获得的随机数在密码学中的应用。
'''

class LCG:
    def __init__(self, seed):
        self.n = seed
        self.A = 1140671485
        self.B = 128201163
        self.M = 2**31

    def extract_number(self):
        # n_next = (A * n + B) % M
        n_ = (self.A * self.n + self.B) % self.M
        self.n = _int32(n_)
    
        return self.n


'''
梅森旋转的测试
'''
def test_MT():
    import time
    seed = int(time.time())
    a = MT19937(seed)
    seed = int(time.time())
    b = MT19937(seed)

    print "a,b extract_number:", a.extract_number(), b.extract_number() 

'''
LCG的测试
'''
def test_LCG():
    import time
    seed = int(time.time())
    a = LCG(seed)
    seed = int(time.time())
    b = LCG(seed)

    print "a,b extract_number:", a.extract_number(), b.extract_number() 


if __name__ == "__main__":
    print "test MT:"
    test_MT()
    
    print "test LCG:"
    test_LCG()
