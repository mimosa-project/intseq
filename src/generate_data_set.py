
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
def generate_dependent_sequence_sample(data:dict, depth:int, numeric_sequence_length=20, is_x_bounded = None)->dict:
    # 数列生成プログラム2つ目作成
    while True: #
        program_gen_2 = generate_program.ProgramGenerator(depth) #
        program_gen_2.build_tree() #

        if is_x_bounded is None: #
            break #
        # is_x_bounded が指定されていて、かつ生成されたプログラムの依存関係が一致する場合にループを抜ける
        elif is_x_bounded == program_gen_2.check_is_x_bounded(): #
            break #
    

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
            'numeric_sequence_2': seq_2[0:20], #
            'token_sequence_2': program_gen_2.get_token_sequence(), #
            'information_amount_2': information_amount_2, #
            'is_x_bounded': int(program_gen_2.check_is_x_bounded()) #
        }
    )

    # 依存数列の順序変更
    if random.random() < 0.5: #
        # スワップ
        temp_seq1 = data['numeric_sequence_1']
        temp_token1 = data['token_sequence_1']
        temp_info1 = data['information_amount_1']

        data['numeric_sequence_1'] = data['numeric_sequence_2']
        data['token_sequence_1'] = data['token_sequence_2']
        data['information_amount_1'] = data['information_amount_2']

        data['numeric_sequence_2'] = temp_seq1
        data['token_sequence_2'] = temp_token1
        data['information_amount_2'] = temp_info1

        data.update({'was_swapped':1}) #
    else:
        data.update({'was_swapped':0}) #
    
    return data #

# 同数の依存数列と非依存数列データの作成
def generate_classification_data(depth:int = None, num_samples=10000): #
    # 指定した深さの数列データを指定数作成
    if depth: #
        # 初期データの生成 (is_x_bounded はまだ含まれない)
        data = [generate_initial_sequence_sample(depth) for _ in range(num_samples)] #

        count_dependent_sequence = 0
        for i in range(len(data)): #
            # 依存関係を持つデータを生成するかどうかの条件
            # count_dependent_sequence < num_samples / 2 は依存関係の数を確認
            # i - count_dependent_sequence < num_samples / 2 は非依存関係の数を確認
            if count_dependent_sequence < num_samples / 2 and i - count_dependent_sequence < num_samples / 2:
                # 依存関係を持つデータを生成するよう指示
                data[i] = generate_dependent_sequence_sample(data[i], depth, is_x_bounded=True) #
            else:
                # 依存関係を持たないデータを生成するよう指示
                data[i] = generate_dependent_sequence_sample(data[i], depth, is_x_bounded=False) #
            
            # data[i] が generate_dependent_sequence_sample の戻り値で更新された後なので、
            # 'is_x_bounded' キーが存在する
            count_dependent_sequence += data[i]['is_x_bounded'] #

    # 指定がなければランダム作成
    else: #
        data = [] #
        for _ in range(num_samples): #
            current_depth = random.randint(3, 15) #
            data.append(generate_initial_sequence_sample(current_depth)) #

        count_dependent_sequence = 0
        for i in range(len(data)): #
            current_depth = random.randint(3, 15) #
            if count_dependent_sequence < num_samples / 2 and i - count_dependent_sequence < num_samples / 2:
                data[i] = generate_dependent_sequence_sample(data[i], current_depth, is_x_bounded=True) #
            else:
                data[i] = generate_dependent_sequence_sample(data[i], current_depth, is_x_bounded=False) #
            
            count_dependent_sequence += data[i]['is_x_bounded'] #
    
    return data #


# コードレビューここまで

'''
data = generate_learning_data(3, 10)
for sample in data:
    if(len(sample['numeric_sequence_1']) != 20 or len(sample['numeric_sequence_2']) != 20):
        print('error')

print(data)
print('finish')
'''