#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random
import math
import sequence_generator

class ProgramGenerator(): 
    # トークンに応じた count の変化量を dict で定義
    POINT_OF_COUNT={'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
    
    def __init__(self):
        self.program_num=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.sequence=[]
        self.information_amount=0 #複雑度
        self.letter_of_choice={}
        self.max_variation_of_program_num=1
    
    def generate_token(self): #rの値に応じてseqにトークンを追加
        self.append_letter()

    def append_letter(self):
        min_variation, max_variation = 1-self.program_num, self.max_variation_of_program_num
        letter, information_amount= self.generate_random_letter(min_variation, max_variation)
        self.sequence.append(letter)
        self.update_program_num(letter)
        self.add_information_amount(information_amount)

    def generate_random_letter(self, min_variation, max_variation):
        if (self.letter_of_choice.get(min_variation, [])):
            return random.choice(self.letter_of_choice[min_variation]), math.log(len(self.letter_of_choice[min_variation]), 2)    
        else:
            letters = []
            letter2variation = {'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
            for k, v in letter2variation.items():
                if min_variation <= v <= max_variation:
                    letters.append(k)
            if(max_variation==1):
                self.letter_of_choice[min_variation]=letters
            return random.choice(letters), math.log(len(letters), 2)
    
    def change_max_variation_of_program_num(self):
        self.max_variation_of_program_num=-1
        self.letter_of_choice={}

    def add_information_amount(self, information_amount):
        self.information_amount+=information_amount
    
    def update_program_num(self, r): #rの値に応じてcountを増減
        self.program_num+=self.VARIATION_OF_PROGRAM_NUM[r] 
    
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
