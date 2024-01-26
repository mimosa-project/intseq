import sys
sys.path.append('../intseq')
from src import program, sequence_generator

def return_sequnece(max_num_of_loops=int, num_of_sequnece=int):
    sequence_list=[]
    sum_of_information_amount=0
    for num in range(num_of_sequnece):
        while(True):
            try:
                sequence=[]
                seq, information_amount = sequence_generator.select_sequence(max_num_of_loops, 10)
                ps=program.ProgramStack(program.ProgramStack.str2rpn(seq), 10)
                stack=ps.build()
                for i in range(10):
                    sequence.append(stack[0].calc(i, 0))
                sequence_list.append(sequence)
                sum_of_information_amount+=information_amount
                break

            except Exception:
                continue
    
    return sequence_list, sum_of_information_amount/num_of_sequnece


max_num_of_loops_20=[]
average_of_information_amount_when_max_num_of_loops_20=0
max_num_of_loops_100=[]
average_of_information_amount_when_max_num_of_loops_100=0
max_num_of_loops_1000=[]
average_of_information_amount_when_max_num_of_loops_1000=0

max_num_of_loops_20, average_of_information_amount_when_max_num_of_loops_20=return_sequnece(20, 100)
max_num_of_loops_100, average_of_information_amount_when_max_num_of_loops_100=return_sequnece(100, 100)
max_num_of_loops_1000, average_of_information_amount_when_max_num_of_loops_1000=return_sequnece(1000, 100)
with open('tests/max_num_of_loops/change2_max_num_of_loops_1000_samples.txt', 'w') as file:
    file.write("max_num_of_loops=20\n")
    for num in range(100):
        file.write(f"{max_num_of_loops_20[num]}\n")
    file.write(f"average of information_amount={average_of_information_amount_when_max_num_of_loops_20}\n")
    file.write("\n")
    file.write("max_num_of_loops=100\n")
    for num in range(100):
        file.write(f"{max_num_of_loops_100[num]}\n")
    file.write(f"average of information_amount={average_of_information_amount_when_max_num_of_loops_100}\n")
    file.write("\n")
    file.write("max_num_of_loops=1000\n")
    for num in range(100):
        file.write(f"{max_num_of_loops_1000[num]}\n")
    file.write(f"average of information_amount={average_of_information_amount_when_max_num_of_loops_1000}\n")
