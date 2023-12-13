#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random as r

class Gen(): 
    def __init__(self):
        self.count=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.seq=[]
        self.r=0
    
    def generate_token(self, r): #rの数に応じて呼び出す関数を限定
        if r<5:
            self.gen1()
        elif r<11:
            self.gen2()
        elif r<13:
            self.gen3()
        elif r==13:
            self.gen5()
    
    def chang_count(self):
        if self.count>4:
            self.r=r.randint(0, 13)
            self.generate_token(self.r)
        elif self.count>2:
            self.r=r.randint(0, 12)
            self.generate_token(self.r)
        elif self.count>1:
            self.r=r.randint(0, 10)
            self.generate_token(self.r)
        else:
            self.gen1()
    
    def reduce_count(self):
        if self.count>4:
            self.r=r.randint(5, 13)
            self.generate_token(self.r)
        elif self.count>2:
            self.r=r.randint(5, 12)
            self.generate_token(self.r)
        elif self.count>1:
            self.gen2()   
    
    #countに対する変化が同じトークンを追加する
    def gen1(self):
        tokens = ['A', 'B', 'C', 'K', 'L']
        self.seq.append(r.choice(tokens))
        self.count += 1

    def gen2(self):
        tokens = ['D', 'E', 'F', 'G', 'H', 'M']
        self.seq.append(r.choice(tokens))
        self.count -= 1

    def gen3(self):
        tokens = ['I', 'J']
        self.seq.append(r.choice(tokens))
        self.count -= 2

    def gen5(self):
        tokens = ['N']
        self.seq.append(r.choice(tokens))
        self.count -= 4

    def count_info(self): #countを出力
        return self.count
    
    def seq_info(self): #seqを出力
        return self.seq

# テスト用翻訳関数
def honyaku(seq):
    for i in range(len(seq)):
        if seq[i]=='A':
            seq[i]='0'
        elif seq[i]=='B':
            seq[i]='1'
        elif seq[i]=='C':
            seq[i]='2'
        elif seq[i]=='D':
            seq[i]='+'
        elif seq[i]=='E':
            seq[i]='-'
        elif seq[i]=='F':
            seq[i]='*'
        elif seq[i]=='G':
            seq[i]='div'
        elif seq[i]=='H':
            seq[i]='mod'
        elif seq[i]=='I':
            seq[i]='cond'
        elif seq[i]=='J':
            seq[i]='loop'
        elif seq[i]=='K':
            seq[i]='x'
        elif seq[i]=='L':
            seq[i]='y'
        elif seq[i]=='M':
            seq[i]='compr'
        elif seq[i]=='N':
            seq[i]='loop2'
    
    return seq

        

def generate():
    seq=Gen()
    n=0
    a=1
    while(a>n):
        seq.chang_count()
        n+=1
        a=r.randint(0, 100)
        
    while(seq.count_info()>1):
        seq.reduce_count()
    
    return seq.seq_info()
