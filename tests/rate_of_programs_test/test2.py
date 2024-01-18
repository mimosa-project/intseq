import src.program as program
import src.sequence_generator as sequence_generator
import matplotlib.pyplot as plt

count_constant_sequence=0
count_trivial_arithmetic_progression=0
num_of_samples=100000

for num in range(num_of_samples):
    while(True):
        try:
            check_if_constant_sequence=0
            check_if_trivial_arithmetic_progression=0
            sequence, information_amount = sequence_generator.generate(100)
            ps=program.ProgramStack(program.ProgramStack.str2rpn(sequence), 10)
            stack=ps.build()
            # yが束縛されていない場合continue
            if not sequence_generator.check_if_y_is_bound(stack):
                continue
            #compr, div, mod, loop, loop2で起きうるエラー持ちの数列を除外
            if sequence_generator.check_if_constant_sequence(stack):
                check_if_constant_sequence=1
            elif sequence_generator.check_if_trivial_arithmetic_progression(stack):
                check_if_trivial_arithmetic_progression=1
            # countに加算
            if check_if_constant_sequence==1:
                count_constant_sequence+=1
            elif check_if_trivial_arithmetic_progression==1:
                count_trivial_arithmetic_progression+=1
            break

        except Exception:
            continue

with open('rate_of_programs_y_is_bound.txt', 'w')as file:
    file.write(f"all samples: {num_of_samples}\nconstant sequence: {count_constant_sequence/num_of_samples}\ntrivial arithmetic progression: {count_trivial_arithmetic_progression/num_of_samples}\nthe others: {(num_of_samples-count_constant_sequence-count_trivial_arithmetic_progression)/num_of_samples}")

labels = ['constant sequence', 'trivial arithmetic progression', 'the others']
sizes = [count_constant_sequence, count_trivial_arithmetic_progression, num_of_samples-count_constant_sequence-count_trivial_arithmetic_progression]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('rate of programs y is bound, 100000 samples')
plt.savefig('rate_of_program_y_is_bound_100000_samples.png')
plt.close()