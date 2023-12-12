#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random as r

class Gen(): 
    def __init__(self):
        self.count=0 #countはリストをオブジェクト化した時の要素数
        self.seq=[] #ProgramStack()へいれるリスト
        self.r=0 #変数
    
    def generate_token(self): #countの数に応じたトークンをseqへ追加
        if self.count>4:
            self.gen5()
        elif self.count>2:
            self.gen3()
        elif self.count>1:
            self.gen2()
        else:
            self.gen1()
    
    def gen1(self): #0,1,2,x,yのトークンのうち1つを等しい確率でsegへ追加
        self.r=r.randint(0,4)
        if(self.r==0):
            self.seq.append('A')
        elif(self.r==1):
            self.seq.append('B')
        elif(self.r==2):
            self.seq.append('C')
        elif(self.r==3):
            self.seq.append('K')
        else:
            self.seq.append('L')
        self.count+=1

    def gen2(self): #0,1,2,x,y,+,-,*,//,%,comprのトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(0, 10)
        if(self.r<5):
            self.gen1()
        elif(self.r==5):
            self.seq.append('D')
            self.count-=1
        elif(self.r==6):
            self.seq.append('E')
            self.count-=1
        elif(self.r==7):
            self.seq.append('F')
            self.count-=1
        elif(self.r==8):
            self.seq.append('G')
            self.count-=1
        elif(self.r==9):
            self.seq.append('H')
            self.count-=1
        elif(self.r==10):
            self.seq.append('M')
            self.count-=1

    def gen3(self): #0,1,2,+,-,*,//,%,compr,cond,loopのトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(0, 12)
        if(self.r<11):
            self.gen2()
        elif(self.r==11):
            self.seq.append('I')
            self.count-=2
        else:
            self.seq.append('J')
            self.count-=2

    def gen5(self): #0,1,2,x,y,+,-,*,//,%,compr,cond,loop,loop2のトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(0, 13)
        if(self.r<13):
            self.gen3()
        else:
            self.seq.append('N')
            self.count-=4
    
    def generate_token_a(self): #+,-,*,//,%,compr,cond,loop,loop2のトークンのみ追加
        if(self.count>4):
            self.gen5_a()
        elif(self.count>2):
            self.gen3_a()
        elif(self.count>1):
            self.gen2_a()
    
    def gen2_a(self): #+,-,*,//,%,comprのトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(5, 10)
        if(self.r==5):
            self.seq.append('D')
            self.count-=1
        elif(self.r==6):
            self.seq.append('E')
            self.count-=1
        elif(self.r==7):
            self.seq.append('F')
            self.count-=1
        elif(self.r==8):
            self.seq.append('G')
            self.count-=1
        elif(self.r==9):
            self.seq.append('H')
            self.count-=1
        elif(self.r==10):
            self.seq.append('M')
            self.count-=1

    def gen3_a(self): #*,//,%,compr,cond,loopのトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(5, 12)
        if(self.r<11):
            self.gen2_a()
        elif(self.r==11):
            self.seq.append('I')
            self.count-=2
        else:
            self.seq.append('J')
            self.count-=2

    def gen5_a(self): #+,-,*,//,%,compr,cond,loop,loop2のトークンのうち1つを等しい確率でseqに追加
        self.r=r.randint(5, 13)
        if(self.r<13):
            self.gen3_a()
        else:
            self.seq.append('N')
            self.count-=4

    def count_info(self): #countを出力
        return self.count
    
    def seq_info(self): #seqを出力
        return self.seq
'''
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
'''
        

def generate():
    seq=Gen()
    n=0
    a=r.randint(1,1)
    while(n<a):
        seq.generate_token()
        n+=1
        a=r.randint(0, 100)
        
    while(seq.seq_info()>1):
        seq.generate_token_a()
    
    return seq.seq_info()
'''
g=generate()
g=honyaku(g)
print(g)
'''
