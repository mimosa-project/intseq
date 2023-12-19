#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random as r
import math

class Gen(): 
    # countの数に対応する呼び出せる文字を randint() でまとめて指定できるように並び替え
    TOKUNS=['A', 'B', 'C', 'K', 'L', 'D', 'E', 'F', 'G', 'H', 'M', 'I', 'J', 'N']
    
    # トークンに応じた count の変化量を dict で定義
    POINT_OF_COUNT={'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
    
    # トークンに応じた comp （トークンの生成される確率の底２の対数のマイナスの総和）の変化量を dict で定義
    POINT_OF_COMP={'A':math.log(5, 2), 'B':math.log(5, 2), 'C':math.log(5, 2), 'K':math.log(5, 2), 'L':math.log(5, 2), 'D':math.log(11, 2), 'E':math.log(11, 2), 'F':math.log(11, 2), 'G':math.log(11, 2), 'H':math.log(11, 2), 'I':math.log(11, 2), 'J':math.log(13, 2), 'M':math.log(13, 2), 'N':math.log(14, 2)}
    
    # generate_token_reduce_count ではトークンの生成される確率が変わる
    MINUSPOINT_OF_COMP={'A':0, 'B':0, 'C':0, 'K':0, 'L':0, 'D':math.log(6, 2), 'E':math.log(6, 2), 'F':math.log(6, 2), 'G':math.log(6, 2), 'H':math.log(6, 2), 'I':math.log(6, 2), 'J':math.log(8, 2), 'M':math.log(8, 2), 'N':math.log(9, 2)}

    def __init__(self):
        self.count=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.seq=[]
        self.comp=0
    
    def generate_token(self, r): #rの値に応じてseqにトークンを追加
        self.seq.append(self.TOKUNS[r])
        self.update_count_and_comp(r)
    
    def update_count_and_comp(self, r): #rの値に応じてcountを増減
        self.count+=self.POINT_OF_COUNT[self.TOKUNS[r]]
        self.comp+=self.POINT_OF_COMP[self.TOKUNS[r]]
    
    def reduce_count_and_update_comp(self, r):
        self.count+=self.POINT_OF_COUNT[self.TOKUNS[r]]
        self.comp+=self.MINUSPOINT_OF_COMP[self.TOKUNS[r]]
    
    def generate_token_change_count(self):
        if (self.count > 4): #オブジェクトが5個以上
            self.generate_token(r.randint(0, 13)) # すべてのトークン
        elif (self.count > 2): #オブジェクトが3個以上
            self.generate_token(r.randint(0, 12)) # loop2を除く
        elif (self.count > 1): #オブジェクトが2個以上
            self.generate_token(r.randint(0, 10)) # cond, loop, loop2を除く
        else: #オブジェクトなし
            self.generate_token(r.randint(0, 4)) # 定数、変数のみ

    def generate_token_reduce_count(self): #change_count()から定数、変数の追加をなくした
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
    
    def comp_info(self):
        return self.comp

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
    
    return seq.seq_info(), seq.comp_info()