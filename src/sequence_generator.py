#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random
import math
import sys
sys.path.append('../intseq')
from src import program


class ProgramGenerator(): 
    # トークンに応じた program_num の変化量を dict で定義
    VARIATION_OF_PROGRAM_NUM={'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
    # generate_random_letterでできたリストをメモ
    MEMO_LETTERS={}

    @classmethod
    def remember_letters(cls, tuple, letters):
        cls.MEMO_LETTERS[tuple]=letters
    
    def __init__(self):
        self.program_num=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.sequence=[]
        self.information_amount=0 #複雑度
    
    def append_letter(self, max_variation=1):
        letter, information_amount= self.generate_random_letter(1-self.program_num, max_variation)
        self.sequence.append(letter)
        self.update_program_num(letter)
        self.add_information_amount(information_amount)

    def generate_random_letter(self, min_variation, max_variation):
        key=(min_variation, max_variation)
        if key in self.MEMO_LETTERS:
            return random.choice(self.MEMO_LETTERS[key]), math.log(len(self.MEMO_LETTERS[key]), 2)
        else:
            letters=[]
            for k, v in self.VARIATION_OF_PROGRAM_NUM.items():
                if min_variation <= v <= max_variation:
                    letters.append(k)
            
            ProgramGenerator.remember_letters(key, letters)
        
            return random.choice(letters), math.log(len(letters), 2)

    def add_information_amount(self, information_amount):
        self.information_amount+=information_amount

    def update_program_num(self, r): #rの値に応じてprogram_numを増減
        self.program_num+=self.VARIATION_OF_PROGRAM_NUM[r]

    def program_num_info(self): # program_num を出力
        return self.program_num
    
    def sequence_info(self): # sequence を出力
        return self.sequence
    
    def information_amount_info(self): # information_amount を出力
        return self.information_amount

# 数列を生成
def generate(max_num_of_loops):
    sequence=ProgramGenerator()
    n=0
    loop_num=random.randint(1, max_num_of_loops)
    while(n < loop_num):
        sequence.append_letter()
        n+=1
        
    # sequneceのprogram_numを1にする
    while(sequence.program_num_info()>1):
        sequence.append_letter(-1)
    
    return sequence.sequence_info(), sequence.information_amount_info()

# 適切な数列を選別
def select_sequence(max_num_of_loops, max_iter=-1):
    while(True):
        try:
            sequence=[]
            information_amount=0
            sequence, information_amount=generate(max_num_of_loops)
            ps=program.ProgramStack(program.ProgramStack.str2rpn(sequence), max_iter)
            stack=ps.build()
            if check_if_y_is_bound(stack) and not check_if_constant_sequence(stack) and not check_if_trivial_arithmetic_progression(stack):
                return sequence, information_amount
            
        except Exception:
            continue

# yが束縛されているかどうかチェック
def check_if_y_is_bound(stack):
    return 'y' not in stack[0].find_free_variables()

# 定数数列かどうかチェック
def check_if_constant_sequence(stack):
    for i in range(1, 10):
        if (stack[0].calc(0, 0) != stack[0].calc(i, 0)):
            return False
    return True

# 初項0、公差1の等差数列(0,1,2,...)かどうかチェック
def check_if_trivial_arithmetic_progression(stack):
    for i in range(0, 10):
        if(stack[0].calc(i, 0) != i):
            return False
    return True
