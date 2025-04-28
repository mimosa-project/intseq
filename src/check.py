import program
import generate_program
import matplotlib.pyplot as plt

def check_sequence(token_sequence):
    counter_dict = {
        'constant_sequence': 0,
        'trivial_arithmetic_progression': 0,
        'arithmetic_progression': 0,
        'other_sequence': 0,
        'sequence_error': 0,
    }

    try:
        program_inter = program.ProgramInterpreter(token_sequence)
        program_executor = program_inter.build()
        if check_if_constant_sequence(program_executor):
            counter_dict['constant_sequence'] += 1
        elif check_if_trivial_arithmetic_progression(program_executor):
            counter_dict['trivial_arithmetic_progression'] += 1
        elif check_if_arithmetic_progression(program_executor):
            counter_dict['arithmetic_progression'] += 1
        else:
            counter_dict['other_sequence'] += 1
    except program.SequenceError as e:
        counter_dict['sequence_error'] += 1
    except Exception as e:
        print(program_executor[0].calc([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))
        print(e)
    finally:
        return counter_dict, program_executor

# 定数数列かどうかチェック
def check_if_constant_sequence(program_executor):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    numeric_sequence = program_executor[0].calc(x)
    for i in range(1, min(len(numeric_sequence),10)):
        if (numeric_sequence[0] != numeric_sequence[i]):
            return False
    return True

# 初項0、公差1の等差数列(0,1,2,...)かどうかチェック
def check_if_trivial_arithmetic_progression(program_executor):
    
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    numeric_sequence = program_executor[0].calc(x)
    for i in range(0, min(len(numeric_sequence),10)):
        if(numeric_sequence[i] != i):
            return False
    return True

# 等差数列かどうかチェック
def check_if_arithmetic_progression(program_executor):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    numeric_sequence = program_executor[0].calc(x)
    a = []
    for i in range(1, min(len(numeric_sequence), 4)):
        a.append(numeric_sequence[i]-numeric_sequence[i-1])
    return all(x == a[0] for x in a)

def generate_sequneces(token_sequence_depth:int, sequnece_num:int):
    counter_dict = {
        'constant_sequence': 0,
        'trivial_arithmetic_progression': 0,
        'arithmetic_progression': 0,
        'other_sequence': 0,
        'sequence_error': 0,
    }

    sequence_program_list = []
    calc_program_list = []
    for num in range(sequnece_num):
        #print(len(sequence_list))
        while(True):
            sequence_program = generate_program.ProgramGenerator(token_sequence_depth)
            sequence_program.build_tree()

            counter_dict_, program_executor = check_sequence(sequence_program.get_token_sequence())
            for k,v in counter_dict_.items():
                assert k in counter_dict_
                counter_dict[k] += v
            sequence_program_list.append(sequence_program)
            calc_program_list.append(program_executor)
            break
    
    return program_list, sequence_list, counter_dict

if __name__ == "__main__":
    program_num=10000
    for token_sequence_depth in range(5,6):
        program_list, sequence_list, counter_dict = generate_sequneces(token_sequence_depth, program_num)
        print("depth:", token_sequence_depth)
        print(counter_dict)
        labels = list(counter_dict.keys())
        values = list(counter_dict.values())
        fig, ax = plt.subplots(figsize=(6, 6))

    # 円グラフを作成
        wedges, texts, aws = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, counterclock=False)

    # ラベルのフォントサイズを変更
        for text in texts:
            text.set_fontsize(20)  
        for aw in aws:
            aw.set_fontsize(20) 
        plt.savefig("pie_chart.png", dpi=300, bbox_inches='tight')

# グラフを表示
plt.show()