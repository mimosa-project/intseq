#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random
import math
import calc

# program_num に応じた生成できるトークンたちを dict型で variation_of_letterに保存、variation_of_letter_reducing_program_num は program_num を減らす場合のもの
lettersvariation = {'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
variation_of_letter={}
variation_of_letter_reducing_program_num={}
for i in range(100):
    letters=[]
    for k, v in lettersvariation.items():    
        if 1-i <= v <= 1:
            letters.append(k)
    variation_of_letter[i]=letters
for i in range(2, 100):
    letters=[]
    for k, v in lettersvariation.items():    
        if 1-i <= v <= -1:
            letters.append(k)
    variation_of_letter_reducing_program_num[i]=letters
class ProgramGenerator(): 
    # トークンに応じた program_num の変化量を dict で定義
    VARIATION_OF_PROGRAM_NUM={'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
    # 上記のプログラムの dict をクラス変数に
    VARIATION_OF_LETTER=variation_of_letter
    VARIATION_OF_LETTER_REDUCING_PROGRAM_NUM=variation_of_letter_reducing_program_num
    
    def __init__(self):
        self.program_num=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.sequence=[]
        self.information_amount=0 #複雑度
        self.max_variation_of_program_num=1
    
    def append_letter(self):
        letter, information_amount= self.generate_random_letter(self.program_num, self.max_variation_of_program_num)
        self.sequence.append(letter)
        self.update_program_num(letter)
        self.add_information_amount(information_amount)

    def generate_random_letter(self, min_variation, max_variation):
        if max_variation==1:
            return random.choice(self.VARIATION_OF_LETTER[min_variation]), math.log(len(self.VARIATION_OF_LETTER[min_variation]), 2)
        else:
            return random.choice(self.VARIATION_OF_LETTER_REDUCING_PROGRAM_NUM[min_variation]), math.log(len(self.VARIATION_OF_LETTER_REDUCING_PROGRAM_NUM[min_variation]), 2)
    
    def change_max_variation_of_program_num(self):
        self.max_variation_of_program_num=-1
        self.letter_of_choice={}

    def add_information_amount(self, information_amount):
        self.information_amount+=information_amount 

    def update_program_num(self, r): #rの値に応じてprogram_numを増減
        self.program_num+=self.VARIATION_OF_PROGRAM_NUM[r]   
    
    def generate_token(self):
        self.append_letter()

    def program_num_info(self): # program_num を出力
        return self.program_num
    
    def sequence_info(self): # sequence を出力
        return self.sequence
    
    def information_amount_info(self): # information_amount を出力
        return self.information_amount

def generate():
    sequence=ProgramGenerator()
    n=0
    a=1
    while(a>n):
        sequence.generate_token()
        n+=1
        a=random.randint(0, 100)
        
    sequence.change_max_variation_of_program_num()
    while(sequence.program_num_info()>1):
        sequence.generate_token()
    
    return sequence.sequence_info(), sequence.information_amount_info()

def sorting():
    while(True):
        try:
            sequence=[]
            information_amount=0
            sequence, information_amount=generate()
            ps=calc.ProgramStack(sequence)
            stack=ps.build()
            if 'y' not in stack[0].find_free_variables():
                if not stack[0].calc(1, 0)==stack[0].calc(10, 0):
                    for num in range(1, 11):
                        print(stack[0].calc(num, 0))
                    break
        except Exception:
            continue

    return sequence, information_amount

for num in range(100):
    g, h=sorting()
    print(calc.ProgramStack.str2rpn(g))

#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2