from src.sequence_generator import *

def test_constant():
    p = Constant(2)
    assert p.calc(1, 2) == 2

def test_variable():
    p1 = Variable('x')
    assert p1.calc(1, 2) == 1

    p2 = Variable('y')
    assert p2.calc(1, 2) == 2

    assert p1.is_free == True
    p1.bind()
    assert p1.is_free == False

def test_cond():
    p1 = Cond(a=Constant(0), b=Constant(1), c=Constant(2))
    assert p1.calc(1, 2) == 1

    p2 = Cond(a=Constant(1), b=Constant(1), c=Constant(2))
    assert p2.calc(1, 2) == 2

def test_loop():
    assert Loop.loop_impl(lambda x,y: x+y, 2, 3) == 6
    assert Loop.loop_impl(lambda x,y: x+y, 3, 2) == 8

    f = Plus(a=Variable('x'), b=Variable('y'))
    p = Loop(f=f, a=Variable('x'), b=Variable('y'))
    assert p.calc(2, 3) == 6
    assert p.calc(3, 2) == 8

def test_loop2():
    assert Loop2.loop2_impl(lambda x,y: x+y, lambda x,y: x*y, 0, 1, 2) == 1
    assert Loop2.loop2_impl(lambda x,y: x+y, lambda x,y: x*y, 1, 1, 2) == 3
    assert Loop2.loop2_impl(lambda x,y: x+y, lambda x,y: x*y, 2, 1, 2) == 5
    assert Loop2.loop2_impl(lambda x,y: x+y, lambda x,y: x*y, 3, 1, 2) == 11
    assert Loop2.loop2_impl(lambda x,y: x+y, lambda x,y: x*y, 4, 1, 2) == 41

    f = Plus(a=Variable('x'), b=Variable('y'))
    g = Multiply(a=Variable('x'), b=Variable('y'))
    p0 = Loop2(f=f, g=g, a=Variable('x'), b=Variable('y'), c=Constant(2))
    assert p0.calc(0, 1) == 1
    p1 = Loop2(f=f, g=g, a=Variable('x'), b=Variable('y'), c=Constant(2))
    assert p1.calc(1, 1) == 3
    p2 = Loop2(f=f, g=g, a=Variable('x'), b=Variable('y'), c=Constant(2))
    assert p2.calc(2, 1) == 5
    p3 = Loop2(f=f, g=g, a=Variable('x'), b=Variable('y'), c=Constant(2))
    assert p3.calc(3, 1) == 11
    p4 = Loop2(f=f, g=g, a=Variable('x'), b=Variable('y'), c=Constant(2))
    assert p4.calc(4, 1) == 41

def test_compr():
    # TODO: implement here
    pass

def test_plus():
    p = Plus(a=Variable('x'), b=Variable('y'))
    assert p.calc(1, 2) == 3

def test_minus():
    m = Minus(a=Variable('x'), b=Variable('y'))
    assert m.calc(1, 2) == -1

def test_multiply():
    m = Multiply(a=Variable('x'), b=Variable('y'))
    assert m.calc(2, 3) == 6

def test_division():
    d = Division(a=Variable('x'), b=Variable('y'))
    assert d.calc(7, 3) == 2

def test_mod():
    m = Mod(a=Variable('x'), b=Variable('y'))
    assert m.calc(7, 3) == 1

def test_program_stack():
    ProgramStack.str2rpn('DFCDCCC') == ['plus', 'multiply', '2', 'plus', '2', '2', '2']

    # Factorial
    ps1 = ProgramStack(['x', 'y', 'multiply', 'x', '1', 'loop'])
    stack = ps1.build()
    assert len(stack) == 1
    stack[0].calc(5,0) == 120
    stack[0].find_free_variables() == {'x'}

    # 2^x (exponential)
    ps2 = ProgramStack(['2', 'x', 'multiply', 'x', '1', 'loop'])
    stack = ps2.build()
    assert len(stack) == 1
    stack[0].calc(5,0) == 32
    stack[0].find_free_variables() == {'x'}

    # Fibonacci
    ps3 = ProgramStack(['x', 'y', 'plus', 'x', 'x', '0', '1', 'loop2'])
    stack = ps3.build()
    assert len(stack) == 1
    stack[0].calc(9,0) == 34
    stack[0].find_free_variables() == {'x'}

    # x^y (power)
    ps4 = ProgramStack(['x', 'y', 'multiply', 'y', 'y', '1', 'x', 'loop2'])
    stack = ps4.build()
    assert len(stack) == 1
    stack[0].calc(2,5) == 32
    stack[0].find_free_variables() == {'x'}
