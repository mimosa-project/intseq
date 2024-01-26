import src.program as program
import src.sequence_generator as sequence_generator
import matplotlib.pyplot as plt

count_y_is_not_bound=0
count_constant_sequence=0
count_trivial_arithmetic_progression=0
num_of_samples=100000

for num in range(num_of_samples):
    while(True):
        try:
            check_if_y_is_bound=0
            check_if_constant_sequence=0
            check_if_trivial_arithmetic_progression=0
            sequence, information_amount = sequence_generator.generate(100)
            ps=program.ProgramStack(program.ProgramStack.str2rpn(sequence), 10)
            stack=ps.build()
            #compr, div, mod, loop, loop2で起きうるエラー持ちの数列を除外
            if not sequence_generator.check_if_y_is_bound(stack):
                check_if_y_is_bound=1
            elif sequence_generator.check_if_constant_sequence(stack):
                check_if_constant_sequence=1
            elif sequence_generator.check_if_trivial_arithmetic_progression(stack):
                check_if_trivial_arithmetic_progression=1
            # countに加算
            if check_if_y_is_bound==1:
                count_y_is_not_bound+=1
            elif check_if_constant_sequence==1:
                count_constant_sequence+=1
            elif check_if_trivial_arithmetic_progression==1:
                count_trivial_arithmetic_progression+=1
            break

        except Exception:
            continue

with open('rate_of_all_programs.txt', 'w')as file:
    file.write(f"all samples: {num_of_samples}\ny is not bound: {count_y_is_not_bound/num_of_samples}\nconstant sequence: {count_constant_sequence/num_of_samples}\ntrivial arithmetic progression: {count_trivial_arithmetic_progression/num_of_samples}\nthe others: {(num_of_samples-count_y_is_not_bound-count_constant_sequence-count_trivial_arithmetic_progression)/num_of_samples}\n\n")
    file.write(f"samples y is bound.\nconstant sequence: {count_constant_sequence/(num_of_samples-count_y_is_not_bound)}\ntrivial arithmetic progression: {count_trivial_arithmetic_progression/(num_of_samples-count_y_is_not_bound)}\nthe others: {(num_of_samples-count_y_is_not_bound-count_constant_sequence-count_trivial_arithmetic_progression)/(num_of_samples-count_y_is_not_bound)}")

labels1 = ['y is not bound', 'constant sequence', 'trivial arithmetic progression', 'the others']
sizes1 = [count_y_is_not_bound, count_constant_sequence, count_trivial_arithmetic_progression, num_of_samples-count_y_is_not_bound-count_constant_sequence-count_trivial_arithmetic_progression]
plt.pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=90)
plt.title('rate of all programs')
plt.savefig('rate_of_all_programs.png')
plt.close()
labels2 = ['constant sequence', 'trivial arithmetic progression', 'the others']
sizes2 = [count_constant_sequence, count_trivial_arithmetic_progression, num_of_samples-count_y_is_not_bound-count_constant_sequence-count_trivial_arithmetic_progression]
plt.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90)
plt.title('rate of programs y is bound')
plt.savefig('rate_of_program_y_is_bound.png')
plt.close()