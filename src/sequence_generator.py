from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Set

class Program:
    def __init__(self, **kwarg:Dict[str, Program]):
        self.fn = None
        self.sub_programs = kwarg
        if 'f' in self.sub_programs:
            self.sub_programs['f'].bind()
        if 'g' in self.sub_programs:
            self.sub_programs['g'].bind()
    
    def build(self) -> Callable[[int, int], int]:
        assert self.fn is not None
        return self.fn
    
    def calc(self, x:int, y:int) -> int:
        return self.build()(x, y)

    def bind(self):
        for p in self.sub_programs.values():
            p.bind()
    
    def find_free_variables(self) -> Set[str]:
        free_variables = set()
        for p in self.sub_programs.values():
            if type(p) is Variable:
                if p.is_free:
                    free_variables.add(p.name)
            else:
                free_variables = free_variables | p.find_free_variables()
        return free_variables
    
class Constant(Program):
    def __init__(self, i):
        super().__init__()
        assert(i in [0,1,2])
        self.i = i

    def build(self) -> Callable[[int, int], int]:
        return self.calc

    def calc(self, x:int, y:int) -> int:
        return self.i
    
class Variable(Program):
    def __init__(self, name):
        super().__init__()
        assert(name in ['x','y'])
        self.name = name
        self.is_free = True
    
    def build(self) -> Callable[[int, int], int]:
        return self.calc
    
    def calc(self, x:int, y:int) -> int:
        if self.name == 'x':
            return x
        else:
            return y
    
    def bind(self):
        self.is_free = False

class Cond(Program):
    @staticmethod
    def cond_impl(a:int, b:int, c:int) -> int:
        if a <= 0:
            return b
        else:
            return c
        
    def build(self) -> Callable[[int, int], int]:
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        fc = self.sub_programs['c'].build()
        
        self.fn = lambda x, y: self.cond_impl(fa(x,y), fb(x,y), fc(x,y))
        return self.fn


class Loop(Program):
    @staticmethod
    def loop_impl(f: Callable[[int, int], int], a: int, b: int) -> int:
        while a > 0:
            b = f(b, a)
            a -= 1
        return b

    def build(self):
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: self.loop_impl(ff, fa(x,y), fb(x,y))
        return self.fn

class Loop2(Program):
    @staticmethod
    def loop2_impl(f: Callable[[int, int], int],
                   g: Callable[[int, int], int],
                   a: int, b: int, c: int) -> int:
        while a > 0:
            b, c = f(b, c), g(b,c)
            a -= 1
        return b

    def build(self) -> Callable[[int, int], int]:
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fg = self.sub_programs['g'].build()
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        fc = self.sub_programs['c'].build()
        
        self.fn = lambda x, y: self.loop2_impl(ff, fg, fa(x,y), fb(x,y), fc(x,y))
        return self.fn

class Compr(Program):
    @staticmethod
    def compr_impl(f:Callable[[int, int], int], a:int, max_iter:int):
        if a == 0:
            m = 0
            while m < max_iter:
                if f(m, 0) <= 0:
                    return m
                m += 1
            raise Exception('compr_impl reached max_iter')
        elif a > 0:
            m = 0
            while m < max_iter:
                if m > Compr.compr_impl(f, a-1, max_iter) and f(m, 0) <= 0:
                    return m
                m += 1
            raise Exception('compr_impl reached max_iter')
        else:
            raise Exception('compr error')

    def build(self):
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fa = self.sub_programs['a'].build()
        
        max_iter=100
        
        self.fn = lambda x, y: self.compr_impl(ff, fa(x,y), max_iter)
        return self.fn
    
class Plus(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) + fb(x,y)
        return self.fn
        
class Minus(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) - fb(x,y)
        return self.fn

class Multiply(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) * fb(x,y)
        return self.fn

class Division(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()

        if lambda x, y: fb(x,y)==0:
            raise Exception("Division by zero")
        self.fn = lambda x, y: fa(x,y) // fb(x,y)
        return self.fn

class Mod(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()

        if lambda x, y: fb(x,y)==0:
            raise Exception("Mod by zero")
        self.fn = lambda x, y: fa(x,y) % fb(x,y)
        return self.fn

class ProgramStack:
    STR2RPN_LIST = ['0', '1', '2', 'plus', 'minus', 'multiply', 'div', 'mod', 'cond', 'loop', 'x', 'y', 'compr', 'loop2']

    def __init__(self, rpn):
        self.rpn = self.str2rpn(rpn)
        self.stack = []
    
    @staticmethod
    def str2rpn(str):
        return [ProgramStack.STR2RPN_LIST[ord(c)-ord('A')] for c in str]

    def build(self):
        for s in self.rpn:
            if isinstance(s, int) or s in ['0', '1', '2']:
                self.stack.append(Constant(int(s)))
            elif s in ['x', 'y']:
                self.stack.append(Variable(s))
            elif s == 'cond':
                c = self.stack.pop()
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Cond(a=a, b=b, c=c))
            elif s == 'loop':
                b = self.stack.pop()
                a = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Loop(f=f, a=a, b=b))
            elif s == 'loop2':
                c = self.stack.pop()
                b = self.stack.pop()
                a = self.stack.pop()
                g = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Loop2(f=f, g=g, a=a, b=b, c=c))
            elif s == 'compr':
                a = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Compr(f=f, a=a))
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
        return self.stack
