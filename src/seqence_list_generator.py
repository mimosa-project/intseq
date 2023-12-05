#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random as r

class Gen():
    def __init__(self):
        self.count=0
        self.seq=[]
        self.r=0
    
    def generate_token(self):
        if self.count>4:
            self.gen5()
        elif self.count>2:
            self.gen3()
        elif self.count>1:
            self.gen2()
        else:
            self.gen1()
    
    def gen1(self):
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

    def gen2(self):
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

    def gen3(self):
        self.r=r.randint(0, 12)
        if(self.r<11):
            self.gen2()
        elif(self.r==11):
            self.seq.append('I')
            self.count-=2
        else:
            self.seq.append('J')
            self.count-=2

    def gen5(self):
        self.r=r.randint(0, 13)
        if(self.r<13):
            self.gen3()
        else:
            self.seq.append('N')
            self.count-=4

    def count_info(self):
        return self.count
    
    def seq_info(self):
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
        
    
    
    return seq.seq_info()
'''
g=generate()
g=honyaku(g)
print(g)
'''
