
import generate_program
import program as program
import token_sequence_converter
import weight

import numpy as np
import random
import string

# 数列データ生成関数
def generate_single_sample_first_sequence(depth:int, numeric_sequence_length=20)->dict:
    # 数列生成プログラム1つ目作成
    pr_1 = generate_program.ProgramGenerator(depth)
    pr_1.build_tree()
    
    # 1つ目の数列生成に必要な数列の長さを測定前処理(トークン列のツリー構造化)
    build_tree_program_1 = token_sequence_converter.TokenTreeConverter(pr_1.get_token_sequence())
    tree_pr_1 = build_tree_program_1.build_tree()
    # 1つ目の数列生成に必要な数列の長さを測定

    # generate_program.calculate_original_sequence_length(numeric_sequence_length, seq_program_tokens)
    # seq_program_tokensによって生成される数列の長さがnumeric_sequence_lengthの場合に、計算する際に代入する数列の必要な長さ（calc()に入力する数列の長さ）を出力
    necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(tree_pr_1, numeric_sequence_length)

    # 1つ目の数列を生成
    x = list(range(necessary_numeric_sequence_length_1))

    calc_1 = program.ProgramStack(pr_1.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_1)
    calc_stack_1 = calc_1.build()
    seq_1 = calc_stack_1[0].calc(x)

    # 1つ目の数列の複雑度
    infomation_amount_1 = pr_1.get_information_amount()
            
    return {
        'sequence_1': seq_1[0:20],
        'program_1': pr_1.get_token_sequence(),
        'infomation_amount_1': infomation_amount_1
    }

# 与えられた数列データに2つ目の数列データを追加する関数
def generate_single_sample_second_sequence(data:dict, depth:int, numeric_sequence_length=20)->dict:
    # 数列生成プログラム2つ目作成
    pr_2 = generate_program.ProgramGenerator(depth)
    pr_2.build_tree()

    # 2つ目の数列生成に必要な数列の長さを測定前処理(トークン列のツリー構造化)
    build_tree_program_2 = token_sequence_converter.TokenTreeConverter(pr_2.get_token_sequence())
    tree_pr_2 = build_tree_program_2.build_tree()
    # 2つ目の数列生成に必要な数列の長さを測定

    necessary_numeric_sequence_length_2 = generate_program.calculate_original_sequence_length(tree_pr_2, numeric_sequence_length)

    # 1つ目の数列生成に必要な数列の長さを測定前処理(トークン列のツリー構造化)
    build_tree_program_1 = token_sequence_converter.TokenTreeConverter(data['program_1'])
    tree_pr_1 = build_tree_program_1.build_tree()

    # 1つ目の数列生成に必要な数列の長さを測定
    necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(tree_pr_1, necessary_numeric_sequence_length_2)

    # 1つ目の数列を生成
    x = list(range(necessary_numeric_sequence_length_1))

    calc_1 = program.ProgramStack(data['program_1'], numeric_sequence_length=necessary_numeric_sequence_length_1)
    calc_stack_1 = calc_1.build()
    seq_1 = calc_stack_1[0].calc(x)

    # 2つ目の数列の生成
    calc_2 = program.ProgramStack(pr_2.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_2)
    calc_stack_2 = calc_2.build()
    seq_2 = calc_stack_2[0].calc(seq_1)

    # 2つ目の数列の複雑度
    infomation_amount_2 = pr_2.get_information_amount()

    # dataに2つ目の数列関連の情報追加
    data.update(
        {
            'sequence_2': seq_2[0:20],
            'program_2': pr_2.get_token_sequence(),
            'infomation_amount_2': infomation_amount_2
        }
    )
    
    return data


def generate_learning_data(depth:int, num_samples=10000):
    # データ生成
    data = [generate_single_sample_first_sequence(depth) for _ in range(num_samples)]

    '''
    print('first_sequence')
    for sample in data:
        print(sample)
        print('sequence_1_length:', len(sample['sequence_1']))
    '''

    for sample in data:
        sample = generate_single_sample_second_sequence(sample, depth)
    
    '''
    print('second_sequence')
    for sample in data:
        print(sample)
        print('sequence_2_length:', len(sample['sequence_2']))
    '''
    
    return data

data = generate_learning_data(3, 10)
for sample in data:
    if(len(sample['sequence_1']) != 20 or len(sample['sequence_2']) != 20):
        print('error')

print(data)
print('finish')