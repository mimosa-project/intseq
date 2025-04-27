from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Set
import math
from sympy.functions.combinatorial.numbers import stirling, catalan
from sympy.ntheory import mobius
import numpy as np

class SequenceError(Exception):
    def __init__(self):
        super().__init__("Insufficient elements for sequence generation.")

class Program:
    def __init__(self, **kwarg:Dict[str, Program]):
        self.sub_programs = kwarg

class Constant(Program):
    def __init__(self, i, seq_length:int =20):
        super().__init__()
        assert(i in [0,1,2])
        self.i = i
        self.seq_length = seq_length

    def calc(self, x: List[int]) -> List[int]:
         return [self.i] * self.seq_length


class Variable(Program):
    def __init__(self, name):
        super().__init__()
        assert(name in ['x'])
        self.name = name
        self.is_free = True
    
    def calc(self, x: List[int]) -> List[int]:
        if self.name == 'x':
            return x


class Plus(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq=[]
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)

        if min(len(seq_a), len(seq_b)) == 0:
            raise SequenceError()
        
        for num in range(min(len(seq_a), len(seq_b))):
            seq.append(seq_a[num] + seq_b[num])

        return seq 

class Minus(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq=[]
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)

        if min(len(seq_a), len(seq_b)) == 0:
            raise SequenceError()
        
        for num in range(min(len(seq_a), len(seq_b))):
            seq.append(seq_a[num] - seq_b[num])

        return seq 

class Multiply(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq=[]
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)

        if min(len(seq_a), len(seq_b)) == 0:
            raise SequenceError()
        
        for num in range(min(len(seq_a), len(seq_b))):
            seq.append(seq_a[num] * seq_b[num])

        return seq 

class Division(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq=[]
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)

        if min(len(seq_a), len(seq_b)) == 0:
            raise SequenceError()
        
        for num in range(min(len(seq_a), len(seq_b))):
            if seq_b[num]==0:
                #print("zero Division")
                seq.append(seq_a[num])
            else:
                seq.append(seq_a[num] // seq_b[num])

        return seq 

class Mod(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq=[]
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)

        if min(len(seq_a), len(seq_b)) == 0:
            raise SequenceError()
        
        for num in range(min(len(seq_a), len(seq_b))):
            if seq_b[num]==0:
                #print("zero Division")
                seq.append(seq_a[num])
            else:
                seq.append(seq_a[num] % seq_b[num])

        return seq 

class Partial_sum(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq = [seq_x[0]]

        for num in range(len(seq_x)-1):
            seq.append(seq[num] + seq_x[num+1])

        return seq

class Partial_sum_of_squares(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq = [seq_x[0]**2]

        for num in range(len(seq_x)-1):
            seq.append(seq[num] + seq_x[num+1]**2)

        return seq

class Self_convolution(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq = []

        if len(seq_x) < 2:
            raise SequenceError()

        for i in range(len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += seq_x[j] * seq_x[i - j]
            seq.append(sum)

        return seq

class Linear_weighted_partial_sums(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq = [0]

        for num in range(1, len(seq_x)):
            seq.append(seq[num-1] + num * seq_x[num])
        
        return seq

class Binomial(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq =[]

        for i in range(len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += math.comb(i, j) * seq_x[j]
            seq.append(sum)
            
        return seq
    
class Inverse_binomial_transform(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq =[]

        for i in range(len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += (-1)**j * math.comb(i, j) * seq_x[j]
            seq.append(sum)
            
        return seq

class Product_of_two_consecutive_elements(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq =[]

        if len(seq_x) < 2:
            raise SequenceError()

        for num in range(seq_x_length-1):
            seq.append(seq_x[num] * seq_x[num+1])
        
        return seq

class Cassini(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq =[]

        if seq_x_length < 3:
            raise SequenceError()

        for num in range(1, seq_x_length-1):
            seq.append(seq_x[num-1] * seq_x[num+1] - seq_x[num]**2)
        
        return seq

class First_stirling(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq =[]

        for i in range(len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += stirling(i, j, kind=1) * seq_x[j]
            seq.append(sum)
        
        return seq

class Second_stirling(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq =[]

        for i in range(len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += stirling(i, j, kind=2) * seq_x[j]
            seq.append(sum)
        
        return seq

class First_differences(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq =[]

        if seq_x_length < 2:
            raise SequenceError()

        for num in range(1, seq_x_length):
            seq.append(seq_x[num] - seq_x[num-1])
        
        return seq

class Catalan(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq =[catalan(0) * seq_x[0]]
        
        for i in range(1, len(seq_x)):
            sum = 0
            for j in range(i+1):
                sum += math.comb(2*i - j - 1, i - j) * j * seq_x[j] // i
            seq.append(sum)

        return seq

class Sum_of_divisors(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq = [0]

        for num in range(1, seq_x_length):
            seq.append(0)
            divs = make_divisors(num)
            
            for divs_num in range(len(divs)):
                seq[num] += seq_x[divs[divs_num]]
        
        return seq

class Moebius(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq =[0]

        for num in range(1, seq_x_length):
            seq.append(0)
            divs = make_divisors(num)
            
            for divs_num in range(len(divs)):
                seq[num] += mobius(num // divs[divs_num]) * seq_x[divs[divs_num]]
        
        return seq

class Hankel(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        seq =[seq_x[0]]

        for num in range(1, math.ceil(seq_x_length/2)):
            hankel = []
            for i in range(num+1):
                hankel.append([])
                for j in range(num+1):
                    hankel[i].append(seq_x[i + j])
            seq.append(self.det(hankel))
        
        return seq
    '''   
    # https://stackoverflow.com/questions/66192894/precise-determinant-of-integer-nxn-matrix 参考
    def det(self, M):
        M = np.array(M, dtype=int) # make a copy to keep original M unmodified
        N, sign, prev = len(M), 1, 1
        for i in range(N-1):
            if M[i, i] == 0: # swap with another row having nonzero i's elem
                swapto = next( (j for j in range(i+1,N) if M[j, i] != 0), None )
                if swapto is None:
                    return 0 # all M[*][i] are zero => zero determinant
                M[[i, swapto]] = M[[swapto, i]]  # Swap rows
                sign = -sign
            for j in range(i+1,N):
                for k in range(i+1,N):
                    assert ( M[j, k] * M[i, i] - M[j, i] * M[i, k] ) % prev == 0
                    M[j, k] = ( M[j, k] * M[i, i] - M[j, i] * M[i, k] ) // prev
            prev = M[i, i]
        return sign * M[-1, -1]
    '''
    def det(self, M):
        N = len(M)
        sign = 1
        
        # 深いコピーを作成して元のMを変更しないようにする
        M = [row[:] for row in M]
        prev = 1  # 最初の分母として使用される値を初期化
        for i in range(N-1):
            if M[i][i] == 0: # swap with another row having nonzero i's elem
                swapto = next( (j for j in range(i+1,N) if M[j][i] != 0), None )
                if swapto is None:
                    return 0 # all M[*][i] are zero => zero determinant
                M[i], M[swapto] = M[swapto], M[i]  # Swap rows
                sign = -sign
            for j in range(i+1,N):
                for k in range(i+1,N):
                    #assert ( M[j, k] * M[i, i] - M[j, i] * M[i, k] ) % prev == 0
                    M[j][k] = ( M[j][k] * M[i][i] - M[j][i] * M[i][k] ) // prev
            prev = M[i][i]
        return sign * M[-1][-1]

class Boustrophedon(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_x = self.sub_programs['a'].calc(x)
        seq_x_length = len(seq_x)
        T=[[seq_x[0]]]
        for i in range(1, seq_x_length):
            T.append([0]*(i+1))
            T[i][0] = seq_x[i]
            for j in range(1, i+1):
                T[i][j] = T[i][j-1] + T[i-1][i-j]
        
        seq=[]

        if seq_x_length < 1:
            raise SequenceError()
        
        for i in range(seq_x_length):
            seq.append(T[i][i])

        return seq

class ProgramStack:
    STR2RPN_LIST = ['0', '1', '2', 'x', 'plus', 'minus', 'multiply', 'division', 'mod', 'partial_sum', 'partial_sum_of_squares', 'self_convolution', 'linear_weighted_partial_sums', 'binomial', 'inverse_binomial_transform', 'product_of_two_consecutive_elements', 'cassini', 'first_stirling', 'second_stirling', 'first_differences', 'catalan', 'sum_of_divisors', 'moebius', 'hankel', 'boustrophedon']

    def __init__(self, rpn, seq_length=20):
        self.rpn = rpn
        self.stack = []
        self.seq_length = seq_length
    
    @staticmethod
    def str2rpn(str):
        return [ProgramStack.STR2RPN_LIST[ord(c)-ord('A')] for c in str]

    def build(self):
        for s in self.rpn:
            if isinstance(s, int) or s in ['0', '1', '2']:
                self.stack.append(Constant(int(s), self.seq_length))
            elif s in ['x']:
                self.stack.append(Variable(s))
            elif s == 'plus':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Plus(a=a, b=b))
            elif s == 'minus':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Minus(a=a, b=b))
            elif s == 'multiply':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Multiply(a=a, b=b))
            elif s == 'division':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Division(a=a, b=b))
            elif s == 'mod':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Mod(a=a, b=b))
            elif s == 'partial_sum':
                a = self.stack.pop()
                self.stack.append(Partial_sum(a=a))
            elif s == 'partial_sum_of_squares':
                a = self.stack.pop()
                self.stack.append(Partial_sum_of_squares(a=a))
            elif s == 'self_convolution':
                a = self.stack.pop()
                self.stack.append(Self_convolution(a=a))
            elif s == 'linear_weighted_partial_sums':
                a = self.stack.pop()
                self.stack.append(Linear_weighted_partial_sums(a=a))
            elif s == 'binomial':
                a = self.stack.pop()
                self.stack.append(Binomial(a=a))
            elif s == 'inverse_binomial_transform':
                a = self.stack.pop()
                self.stack.append(Inverse_binomial_transform(a=a))
            elif s == 'product_of_two_consecutive_elements':
                a = self.stack.pop()
                self.stack.append(Product_of_two_consecutive_elements(a=a))
            elif s == 'cassini':
                a = self.stack.pop()
                self.stack.append(Cassini(a=a))
            elif s == 'first_stirling':
                a = self.stack.pop()
                self.stack.append(First_stirling(a=a))
            elif s == 'second_stirling':
                a = self.stack.pop()
                self.stack.append(Second_stirling(a=a))
            elif s == 'first_differences':
                a = self.stack.pop()
                self.stack.append(First_differences(a=a))
            elif s == 'catalan':
                a = self.stack.pop()
                self.stack.append(Catalan(a=a))
            elif s == 'sum_of_divisors':
                a = self.stack.pop()
                self.stack.append(Sum_of_divisors(a=a))
            elif s == 'moebius':
                a = self.stack.pop()
                self.stack.append(Moebius(a=a))
            elif s == 'hankel':
                a = self.stack.pop()
                self.stack.append(Hankel(a=a))
            elif s == 'boustrophedon':
                a = self.stack.pop()
                self.stack.append(Boustrophedon(a=a))
        return self.stack

# 約数を列挙する関数
def make_divisors(n):
    lower_divisors , upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]
'''
x = [1]
pr = ProgramStack(['B', 'Y'])
stack = pr.build()
print(stack[0].calc(x))
'''
# Hankel() det のテスト
'''
test = Hankel()

test_1 = [[-1, 1, 6, 8],[9,1,0,9],[2,0,0,4],[0,0,0,3]]
# -36
test_2 = [[7,0,7,0],[3,8,3,5],[1,9,1,2],[2,4,2,1]]
# 0
test_3 = [[3,1,5,2],[0,1,2,1],[0,1,1,1],[0,0,2,1]]
# -3
test_4 = [[1,0,0,6],[0,2,3,0],[5,3,0,1],[0,0,5,2]]
# 272
test_5 = [[2,0,9,1],[3,2,5,3],[0,1,1,1],[0,1,5,1]]
# -4
test_6 = [[0,3,-4],[1,0,2],[-1,0,2]]
# -12
test_7 = [[0,0,0,1],[0,0,2,0],[0,3,0,0],[4,0,0,0]]
# 24
test_8 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# 0
test_9 = [[1,-1,1,-1],[1,-2,1,-2],[1,-2,2,-1],[1,-2,2,-2]]
# 1
test_10 = [[4,-1,-2], [0,5,-3],[2,1,2]]
# 78
test_11 = [[0,0,1,1,1],[0,1,1,1,0],[1,1,1,0,0],[1,1,0,0,1],[1,0,0,1,1]]
# 3
print(test.det(test_1))
print(test.det(test_2))
print(test.det(test_3))
print(test.det(test_4))
print(test.det(test_5))
print(test.det(test_6))
print(test.det(test_7))
print(test.det(test_8))
print(test.det(test_9))
print(test.det(test_10))
print(test.det(test_11))
'''
#部分和のテスト
'''
test_1 = [1,2,3]
# [1,3,6]
test_2 = [4,5,6]
# [4,9,15]
test_3 = [0,-1,2]
# [0,-1,1]
test_4 = [10,20,30]
# [10,30,60]
test_5 = [-2,-3,5]
# [-2,-5,0]
test_6 = [1,1,1,1]
#[1,2,3,4]
test_7 = [5,5,5,5]
#[5,10,15,20]
test_8 = [1,-1,1,-1]
#[1,0,1,0]
test_9 = [2,4,6,8]
#[2,6,12,20]
test_10 = [7,3,-2,4]
#[7,10,8,12]
test = ProgramStack(['D','J'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
#自己畳み込みのテストT
'''
test_1 = [1,2,3]
#[1,4,10]
test_2 = [1,1,1]
#[1,2,3]
test_3 = [1,-1]
#[1,-2]
test_4 = [2,3]
#[4,12]
test_5 = [1,0,-1]
#[1,0,-2]
test_6 = [1,2,1]
#[1,4,6]
test_7 = [3,1]
#[9,6]
test_8 = [1,2,3,4]
#[1,4,10,20]
test_9 = [0,1,0]
#[0,0,1]
test_10 = [1,2,1,2]
#[1,4,6,8]
test = ProgramStack(['D','L'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''

#重み付き部分和のテストT
'''
test_1 = [1, 2, 3]
#[0, 2, 8]
test_2 = [4, 5, 6]
#[0, 5, 17]
test_3 = [1, -1, 2]
#[0, -1, 3]
test_4 = [3, 0, 2, -1]
#[0, 0, 4, 1]
test_5 = [1, 2, 1, 2]
#[0, 2, 4, 10]
test_6 = [0, 1, 2, 3]
#[0, 1, 5, 14]
test_7 = [2, 3, 4]
#[0, 3, 11]
test_8 = [1, 2, 3, 4, 5]
#[0, 2, 8, 20, 40]
test_9 = [1, -1, 1, -1, 1]
#[0, -1, 1, -2, 2]
test_10 = [2, 2, 2, 2]
#[0, 2, 6, 12]
test = ProgramStack(['D','M'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
#BinomialのテストT
'''
test_1 = [0,1,2]
#[0,1,4]
test_2 = [1,1,1]
#[1,2,4]
test_3 = [3,2,1]
#[3,5,8]
test_4 = [2,3,1]
#[2,5,9]
test_5 = [0,1,2,3]
#[0,1,4,12]
test_6 = [3,2,1,0]
#[3,5,8,12]
test_7 = [1,1,1,1]
#[1,2,4,8]
test_8 = [2,0,3,1]
#[2,2,5,12]
test_9 = [0,1,2,3,4]
#[0,1,4,12,32]
test_10 = [1,1,1,1,1]
#[1,2,4,8,16]
test = ProgramStack(['D', 'N'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
for i in range(1):
    for j in range(i+1):
        print(i,j)
        print(math.comb(i,j))
'''
#L_BinomialのテストT
'''
test_1 = [0,1,2]
#[0,-1,0]
test_2 = [1,1,1]
#[1,0,0]
test_3 = [3,2,1]
#[3,1,0]
test_4 = [2,3,1]
#[2,-1,-3]
test_5 = [0,1,2,3]
#[0,-1,0,0]
test_6 = [3,2,1,0]
#[3,1,0,0]
test_7 = [1,1,1,1]
#[1,0,0,0]
test_8 = [2,0,3,1]
#[2,2,5,10]
test_9 = [0,1,2,3,4]
#[0,-1,0,0,0]
test_10 = [1,1,1,1,1]
#[1,0,0,0,0]
test = ProgramStack(['D', 'O'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
for i in range(5):
    for j in range(i+1):
        print(i,j)
        print((-1)**j*math.comb(i,j))
'''
# 連続項の積のテストT
'''
test_1 = [0,1,2]
#[0,2]
test_2 = [1,1,1]
#[1,1]
test_3 = [3,2,1]
#[6,2]
test_4 = [2,3,1]
#[6,3]
test_5 = [0,1,2,3]
#[0,2,6]
test_6 = [3,2,1,0]
#[6,2,0]
test_7 = [1,2,1,2]
#[2,2,2]
test_8 = [2,0,3,1]
#[0,0,3]
test_9 = [0,1,2,3,4]
#[0,2,6,12]
test_10 = [10,5,3,2,1]
#[50,15,6,2]
test = ProgramStack(['D', 'P'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
# cassiniのテストT
'''
test_1 = [0,1,2]
#[-1]
test_2 = [1,1,1]
#[0]
test_3 = [3,2,1]
#[-1]
test_4 = [2,3,1]
#[-7]
test_5 = [1,1,2,3]
#[1,-1]
test_6 = [3,1,3,5]
#[8,-4]
test_7 = [1,2,1,2]
#[-3,3]
test_8 = [2,0,3,1]
#[6,-9]
test_9 = [3,1,2,2,4]
#[5,-2,4]
test_10 = [10,5,3,2,1]
#[5,1,-1]
test = ProgramStack(['D', 'Q'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
# first_stirlingのテストT
'''
test_1 = [0,1,2]
#[0,1,3]
test_2 = [1,1,1]
#[1,1,2]
test_3 = [3,2,1]
#[3,2,3]
test_4 = [2,3,1]
#[2,3,4]
test_5 = [1,1,2,3]
#[1,1,3,12]
test_6 = [3,1,3,5]
#[3,1,4,16]
test_7 = [1,2,1,2]
#[1,2,3,9]
test_8 = [2,0,3,1]
#[2,0,3,10]
test_9 = [3,1,2,2,4]
#[3,1,3,10,44]
test_10 = [10,5,3,2,1]
#[10,5,8,21,76]
test = ProgramStack(['D', 'R'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
for i in range(5):
    for j in range(i+1):
        print(i,j)
        print(stirling(i,j,kind=1))
'''
# second_stirlingのテストT
'''
test_1 = [0,1,2]
#[0,1,3]
test_2 = [1,1,1]
#[1,1,2]
test_3 = [3,2,1]
#[3,2,3]
test_4 = [2,3,1]
#[2,3,4]
test_5 = [1,1,2,3]
#[1,1,3,10]
test_6 = [3,1,3,5]
#[3,1,4,15]
test_7 = [1,2,1,2]
#[1,2,3,7]
test_8 = [2,0,3,1]
#[2,0,3,10]
test_9 = [3,1,2,2,4]
#[3,1,3,9,31]
test_10 = [10,5,3,2,1]
#[10,5,8,16,39]
test = ProgramStack(['D', 'S'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
for i in range(5):
    for j in range(i+1):
        print(i,j)
        print(stirling(i,j,kind=2))
'''
# first_differencesのテストT
'''
test_1 = [0,1,2]
#[1,1]
test_2 = [1,1,1]
#[0,0]]
test_3 = [3,2,1]
#[-1,-1]
test_4 = [2,3,1]
#[1,-2]
test_5 = [1,1,2,3]
#[0,1,1]
test_6 = [3,1,3,5]
#[-2,2,2]
test_7 = [1,2,1,2]
#[1,-1,1]
test_8 = [2,0,3,1]
#[-2,3,-2]
test_9 = [3,1,2,2,4]
#[-2,1,0,2]
test_10 = [10,5,3,2,1]
#[-5,-2,-1,-1]
test = ProgramStack(['D', 'T'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
# catalanのテストT
'''
test_1 = [0,1,2]
#[0,1,3]
test_2 = [1,1,1]
#[1,1,2]
test_3 = [3,2,1]
#[3,2,3]
test_4 = [2,3,1]
#[2,3,4]
test_5 = [1,1,2,3]
#[1,1,3,9]
test_6 = [3,1,3,5]
#[3,1,4,13]
test_7 = [1,2,1,2]
#[1,2,3,8]
test_8 = [2,0,3,1]
#[2,0,3,7]
test_9 = [3,1,2,2,4]
#[3,1,3,8,25]
test_10 = [10,5,3,2,1]
#[10,5,8,18,47]
test = ProgramStack(['D', 'U'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
for i in range(1, 5):
    for j in range(i+1):
        print(i,j)
        print((math.comb(2 * i - j - 1 , i - j)*j//i))
'''
# 約数項の総和のテスト
'''
test_1 = [0, 1, 2, 3, 4, 5, 6]
#[0, 1, 3, 4, 7, 6, 12]
test_2 = [0, 2, 3, 5, 7, 11, 13]
#[0, 2, 5, 7, 12, 13, 23]
test_3 = [0, 5, 10, 15, 20, 25, 30]
#[0, 5, 15, 20, 35, 30, 60]
test_4 = [0, 1, 4, 9, 16, 25, 36]
#[0, 1, 5, 10, 21, 26, 50]
test_5 = [0, 8, 16, 24, 32, 40, 48]
#[0, 8, 24, 32, 56, 48, 96]
test_6 = [0, 7, 14, 21, 28, 35, 42]
#[0, 7, 21, 28, 49, 42, 84]
test_7 = [0, 3, 6, 9, 12, 15, 18]
#[0, 3, 9, 12, 21, 18, 36]
test_8 = [0, 4, 8, 12, 16, 20, 24]
#[0, 4, 12, 16, 28, 24, 48]
test_9 = [0, 9, 18, 27, 36, 45, 54]
#[0, 9, 27, 36, 63, 54, 108]
test_10 = [1,1,1,1,1,1,1]
#[0,1,2,2,3,2,4]
test = ProgramStack(['D', 'V'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
# moebiusT
'''
test_1 = [0,1,2]
#[0,1,1]
test_2 = [1,1,1]
#[0,1,0]
test_3 = [3,2,1]
#[0,2,-1]
test_4 = [2,3,1]
#[0,3,-2]
test_5 = [1,1,2,3]
#[0,1,1,2]
test_6 = [3,1,3,5]
#[0,1,2,4]
test_7 = [1,2,1,2]
#[0,2,-1,0]
test_8 = [2,0,3,1]
#[0,0,3,1]
test_9 = [3,1,2,2,4]
#[0,1,1,1,2]
test_10 = [10,5,3,2,1]
#[0,5,-2,-3,-2]
test = ProgramStack(['D', 'W'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
print("1,1")
print(mobius(1))
print("2,1")
print(mobius(2))
print("2,2")
print(mobius(1))
print("3,1")
print(mobius(3))
print("3,3")
print(mobius(1))
print("4,1")
print(mobius(4))
print("4,2")
print(mobius(2))
print("4,4")
print(mobius(1))
'''
# hankelのテストT
'''
test_1 = [0,1,2]
test_2 = [1,1,1]
test_3 = [3,2,1]
test_4 = [2,3,1]
test_5 = [1,1,2,3]
test_6 = [3,1,3,5]
test_7 = [1,2,1,2]
test_8 = [2,0,3,1]
test_9 = [3,1,2,2,4]
test_10 = [10,5,3,2,1]
test = ProgramStack(['D', 'X'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
# boustrophedonのテストT
'''
test_1 = [0,1,2]
#[0,1,4]
test_2 = [1,1,1]
#[1,2,4]
test_3 = [3,2,1]
#[3,5,8]
test_4 = [2,3,1]
#[2,5,9]
test_5 = [1,1,2,3]
#[1,2,5,14]
test_6 = [3,1,3,5]
#[3,4,8,23]
test_7 = [1,2,1,2]
#[1,3,6,13]
test_8 = [2,0,3,1]
#[2,2,5,14]
test_9 = [3,1,2,2,4]
#[3,4,7,17,47]
test_10 = [10,5,3,2,1]
#[10,15,23,46,117]
test = ProgramStack(['D', 'Y'])
stack = test.build()
print(stack[0].calc(test_1))
print(stack[0].calc(test_2))
print(stack[0].calc(test_3))
print(stack[0].calc(test_4))
print(stack[0].calc(test_5))
print(stack[0].calc(test_6))
print(stack[0].calc(test_7))
print(stack[0].calc(test_8))
print(stack[0].calc(test_9))
print(stack[0].calc(test_10))
'''
