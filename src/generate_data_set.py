
import generate_program
import program as program
import weight

import numpy as np
import random
import string

# コードレビューここから
# 数列データ生成関数
def generate_initial_sequence_sample(depth:int, numeric_sequence_length=20)->dict:
    # 数列生成プログラム1つ目作成
    program_gen_1 = generate_program.ProgramGenerator(depth)
    program_gen_1.build_tree()
    
    # 1つ目の数列生成に必要な数列の長さを測定

    # generate_program.calculate_original_sequence_length(tree_program.root_node, numeric_sequence_length)
    # tree_programによって生成される数列の長さがnumeric_sequence_lengthの場合に、計算する際に代入する数列の必要な長さ（calc()に入力する数列の長さ）を出力
    necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1.root_node, numeric_sequence_length)

    # 1つ目の数列を生成
    x = list(range(necessary_numeric_sequence_length_1))

    program_inter_1 = program.ProgramInterpreter(program_gen_1.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_1)
    program_executor_1 = program_inter_1.build()
    seq_1 = program_executor_1[0].calc(x)

    # 1つ目の数列の複雑度
    information_amount_1 = program_gen_1.get_information_amount()
            
    return {
        'numeric_sequence_1': seq_1[0:20],
        'token_sequence_1': program_gen_1.get_token_sequence(),
        'information_amount_1': information_amount_1
    }

# 与えられた数列データに2つ目の数列データを追加する関数
def generate_dependent_sequence_sample(data:dict, depth:int, numeric_sequence_length=20)->dict:
    # 数列生成プログラム2つ目作成
    program_gen_2 = generate_program.ProgramGenerator(depth)
    program_gen_2.build_tree()

    # 2つ目の数列生成に必要な数列の長さを測定

    necessary_numeric_sequence_length_2 = generate_program.calculate_original_sequence_length(program_gen_2.root_node, numeric_sequence_length)

    # 1つ目の数列生成に必要な数列の長さを測定前処理(トークン列のツリー構造化)
    program_gen_1 = generate_program.ProgramGenerator(depth)
    program_gen_1.convert_token_sequence_to_tree(data['token_sequence_1'])

    # 1つ目の数列生成に必要な数列の長さを測定
    necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1.root_node, necessary_numeric_sequence_length_2)

    # 1つ目の数列を生成
    x = list(range(necessary_numeric_sequence_length_1))

    program_inter_1 = program.ProgramInterpreter(data['token_sequence_1'], numeric_sequence_length=necessary_numeric_sequence_length_1)
    program_executor_1 = program_inter_1.build()
    seq_1 = program_executor_1[0].calc(x)

    # 2つ目の数列の生成
    program_inter_2 = program.ProgramInterpreter(program_gen_2.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_2)
    program_executor_2 = program_inter_2.build()
    seq_2 = program_executor_2[0].calc(seq_1)

    # 2つ目の数列の複雑度
    information_amount_2 = program_gen_2.get_information_amount()

    # dataに2つ目の数列関連の情報追加
    data.update(
        {
            'numeric_sequence_2': seq_2[0:20],
            'token_sequence_2': program_gen_2.get_token_sequence(),
            'information_amount_2': information_amount_2
        }
    )
    
    return data


def generate_learning_data(depth:int = None, num_samples=10000):
    # 指定した深さの数列データを指定数作成
    if depth:
        data = [generate_initial_sequence_sample(depth) for _ in range(num_samples)]

        for i in range(len(data)):
            data[i] = generate_dependent_sequence_sample(data[i], depth)
    # 指定がなければランダム作成
    else:
        depth = random.randint(3, 15)
        data = [generate_initial_sequence_sample(depth) for _ in range(num_samples)]

        depth = random.randint(3, 15)
        for i in range(len(data)):
            data[i] = generate_dependent_sequence_sample(data[i], depth)
    
    
    return data

# コードレビューここまで

'''
data = generate_learning_data(3, 10)
for sample in data:
    if(len(sample['numeric_sequence_1']) != 20 or len(sample['numeric_sequence_2']) != 20):
        print('error')

print(data)
print('finish')
'''