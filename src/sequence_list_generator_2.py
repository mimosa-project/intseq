#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random as r

class Gen(): 
    def __init__(self):
        self.count=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.seq=[]
        self.tokens=['A', 'B', 'C', 'K', 'L', 'D', 'E', 'F', 'G', 'H', 'M', 'I', 'J', 'N']
    
    def generate_token(self, r): #rの値に応じてseqにトークンを追加
        self.seq.append(self.tokens[r])
        self.update_count(r)
    
    def update_count(self, r): #rの値に応じてcountを増減
        if(r==13):
            self.count-=4
        elif(r>10):
            self.count-=2
        elif(r>4):
            self.count-=1
        else:
            self.count+=1
    
    def change_count(self):
        if (self.count > 4): #オブジェクトが5個以上
            self.generate_token(r.randint(0, 13)) # すべてのトークン
        elif (self.count > 2): #オブジェクトが3個以上
            self.generate_token(r.randint(0, 12)) # loop2を除く
        elif (self.count > 1): #オブジェクトが2個以上
            self.generate_token(r.randint(0, 10)) # cond, loop, loop2を除く
        else: #オブジェクトなし
            self.generate_token(r.randint(0, 4)) # 定数、変数のみ

    def reduce_count(self): #change_count()から定数、変数の追加をなくした
        if (self.count > 4):
            self.generate_token(r.randint(5, 13)) 
        elif (self.count > 2):
            self.generate_token(r.randint(5, 12)) 
        elif (self.count > 1):
            self.generate_token(r.randint(5, 10)) 

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
        seq.change_count()
        n+=1
        a=r.randint(0, 100)
        
    while(seq.count_info()>1):
        seq.reduce_count()
    
    return seq.seq_info()
