import generate_program
import program as program
import weight

import numpy as np
import random
import string
import sys # sysモジュールを追加


# 数列データ生成関数
def generate_initial_sequence_sample(depth:int, numeric_sequence_length=20)->dict:
    """
    1つ目の数列データと関連情報を生成します。
    オーバーフロー、NaN/Inf、定数列、等差数列の場合は再試行します。
    """
    max_attempts = 1000 # 無効なサンプルが生成された場合の最大試行回数
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            program_gen_1 = generate_program.ProgramGenerator(depth)
            program_gen_1.build_tree()
            
            # 1つ目の数列生成に必要な数列の長さを測定
            # calculate_original_sequence_lengthはprogram.pyのcalculate_original_sequence_lengthのこと
            necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1.root_node, numeric_sequence_length)

            # 1つ目の数列を生成する初期値 (x)
            # 0から始まるrangeは定数列になりやすいため、1から開始することも検討
            x = list(range(1, necessary_numeric_sequence_length_1 + 1)) 

            program_inter_1 = program.ProgramInterpreter(program_gen_1.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_1)
            program_executor_1 = program_inter_1.build()
            
            # 数列の計算を実行
            seq_1 = program_executor_1[0].calc(x) 
            
            # --- 数列の品質チェック ---
            # 1. NaN/Inf 値のチェック (浮動小数点数への変換前にOverflowErrorを捕捉しているが、念のため)
            if not seq_1 or any(np.isinf(val) or np.isnan(val) for val in seq_1):
                raise program.SequenceError("Generated sequence 1 is empty or contains Inf/NaN.")

            # 2. 等差数列のチェック (最終的な20要素でチェック)
            if program.check_if_arithmetic_progression(seq_1[0:20]):
                raise program.SequenceError("Generated sequence 1 is an arithmetic progression.")

            # 1つ目の数列の複雑度
            information_amount_1 = program_gen_1.get_information_amount()
            
            return {
                'numeric_sequence_1': seq_1[0:20], # 最終的に必要な20要素にスライス
                'token_sequence_1': program_gen_1.get_token_sequence(),
                'information_amount_1': information_amount_1
            }
        except program.SequenceError as e:
            # print(f"DEBUG: Skipping initial sequence generation due to {e} (Attempt {attempt_count}/{max_attempts})")
            continue # 無効なサンプルなので再試行
        except Exception as e: # その他の予期せぬエラーも捕捉（例: list index out of rangeなど）
            # print(f"DEBUG: Skipping initial sequence generation due to unexpected error: {type(e).__name__}: {e} (Attempt {attempt_count}/{max_attempts})")
            continue
    
    # 試行回数上限に達しても有効なサンプルが生成できなかった場合
    print(f"ERROR: Failed to generate a valid initial sequence after {max_attempts} attempts. Check program generation logic.")
    return None # 無効なサンプルとしてNoneを返す

# 与えられた数列データに2つ目の数列データを追加する関数
def generate_dependent_sequence_sample(data:dict, depth:int, numeric_sequence_length=20, is_x_bounded = None)->dict:
    """
    与えられた1つ目の数列データに、2つ目の数列データと依存関係情報を追加します。
    オーバーフロー、NaN/Inf、定数列、等差数列の場合は再試行します。
    """
    max_attempts = 1000 # 無限ループ回避のための試行回数上限
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            # 数列生成プログラム2つ目作成
            program_gen_2 = generate_program.ProgramGenerator(depth)
            program_gen_2.build_tree()

            generated_is_x_bounded = program_gen_2.check_is_x_bounded()

            # 要求された依存関係の有無と、生成されたプログラムの依存関係の有無が一致しない場合は再試行
            if is_x_bounded is not None and is_x_bounded != generated_is_x_bounded:
                continue 

            # 2つ目の数列生成に必要な数列の長さを測定
            necessary_numeric_sequence_length_2 = generate_program.calculate_original_sequence_length(program_gen_2.root_node, numeric_sequence_length)

            # 1つ目の数列生成に必要な数列の長さを測定 (トークン列のツリー構造化を含む)
            # generate_program.ProgramGenerator は新しいインスタンスでtreeを再構築する必要があるため、一時的に使用
            program_gen_1_temp = generate_program.ProgramGenerator(depth) 
            program_gen_1_temp.convert_token_sequence_to_tree(data['token_sequence_1'])
            necessary_numeric_sequence_length_1 = generate_program.calculate_original_sequence_length(program_gen_1_temp.root_node, necessary_numeric_sequence_length_2)

            # 1つ目の数列を生成する初期値 (x)
            x_initial = list(range(1, necessary_numeric_sequence_length_1 + 1)) 
            
            # 1つ目の数列を再生成
            program_inter_1 = program.ProgramInterpreter(data['token_sequence_1'], numeric_sequence_length=necessary_numeric_sequence_length_1)
            program_executor_1 = program_inter_1.build()
            seq_1 = program_executor_1[0].calc(x_initial) 

            # 2つ目の数列の生成 (seq_1 を入力として使用)
            program_inter_2 = program.ProgramInterpreter(program_gen_2.get_token_sequence(), numeric_sequence_length=necessary_numeric_sequence_length_2)
            program_executor_2 = program_inter_2.build()
            seq_2 = program_executor_2[0].calc(seq_1) # seq_1をxとしてcalcに渡す
            
            # --- 数列の品質チェック (seq_2) ---
            # 1. NaN/Inf 値のチェック
            if not seq_2 or any(np.isinf(val) or np.isnan(val) for val in seq_2):
                 raise program.SequenceError("Generated sequence 2 is empty or contains Inf/NaN.")
            # 2. 等差数列のチェック
            if program.check_if_arithmetic_progression(seq_2[0:20]):
                raise program.SequenceError("Generated sequence 2 is an arithmetic progression.")

            information_amount_2 = program_gen_2.get_information_amount()

            # data辞書を更新 (元のdataオブジェクトを変更しないようにコピー)
            data_updated = data.copy() 
            data_updated.update(
                {
                    'numeric_sequence_2': seq_2[0:20], # 最終的に必要な20要素にスライス
                    'token_sequence_2': program_gen_2.get_token_sequence(),
                    'information_amount_2': information_amount_2,
                    'is_x_bounded': int(generated_is_x_bounded) # 0または1に変換
                }
            )

            # 依存数列の順序変更 (ランダムにseq1とseq2をスワップ)
            if random.random() < 0.5:
                # スワップ処理は、キー名を変更しつつ、値も入れ替える
                temp_numeric_sequence_1 = data_updated['numeric_sequence_1']
                temp_token_sequence_1 = data_updated['token_sequence_1']
                temp_information_amount_1 = data_updated['information_amount_1']

                data_updated['numeric_sequence_1'] = data_updated['numeric_sequence_2']
                data_updated['token_sequence_1'] = data_updated['token_sequence_2']
                data_updated['information_amount_1'] = data_updated['information_amount_2']

                data_updated['numeric_sequence_2'] = temp_numeric_sequence_1
                data_updated['token_sequence_2'] = temp_token_sequence_1
                data_updated['information_amount_2'] = temp_information_amount_1

                data_updated.update({'was_swapped':1})
            else:
                data_updated.update({'was_swapped':0})
            
            return data_updated # 正常に生成できた場合はここで返す

        except program.SequenceError as e:
            # print(f"DEBUG: Retrying dependent sequence generation due to {e} (Attempt {attempt_count}/{max_attempts})")
            continue # 無効なサンプルなので再試行
        except Exception as e: # その他の予期せぬエラーも捕捉
            # print(f"DEBUG: Retrying dependent sequence generation due to unexpected error: {type(e).__name__}: {e} (Attempt {attempt_count}/{max_attempts})")
            continue
    
    # 試行回数上限に達しても有効なサンプルが生成できなかった場合
    print(f"ERROR: Failed to generate a valid dependent sequence after {max_attempts} attempts for initial data: {data['token_sequence_1']}. Check program generation logic.")
    return None # 無効なサンプルとしてNoneを返す

# 同数の依存数列と非依存数列データの作成 (メインのデータ生成関数)
def generate_classification_data(depth:int = None, num_samples=10000):
    """
    指定された深さまたはランダムな深さで、依存関係のある/ない数列ペアデータセットを生成します。
    無効な（オーバーフロー、NaN/Inf、定数列、等差数列の）サンプルはスキップされ、目標のサンプル数を生成します。
    """
    print(f"[DEBUG] Starting generate_classification_data for {num_samples} samples.")
    
    generated_data = []
    current_dependent_count = 0
    current_independent_count = 0
    
    # 目標のサンプル数を生成するまでループ
    sample_generation_attempts = 0 # 全体での試行回数（無限ループ防止用）
    max_total_attempts = num_samples * 5 # 例: 目標サンプル数の5倍の試行で打ち切る
    
    while len(generated_data) < num_samples and sample_generation_attempts < max_total_attempts:
        sample_generation_attempts += 1
        
        # 1つ目の数列を生成
        if depth:
            initial_sample = generate_initial_sequence_sample(depth)
        else:
            initial_sample = generate_initial_sequence_sample(random.randint(3, 8)) # 深さの上限を8に設定
        
        if initial_sample is None: # initial_sample生成に失敗した場合
            # print(f"DEBUG: Initial sample generation failed. Retrying main loop. (Attempt {sample_generation_attempts})")
            continue

        # 依存/非依存のバランスを考慮して2つ目の数列の生成タイプを決定
        # 目標の依存/非依存のサンプル数が両方とも不足している場合、ランダムに選択
        if current_dependent_count < num_samples / 2 and current_independent_count < num_samples / 2:
            is_bounded_request = random.choice([True, False])
        elif current_dependent_count < num_samples / 2:
            is_bounded_request = True # 依存が不足しているため、依存ペアを生成要求
        else:
            is_bounded_request = False # 非依存が不足しているため、非依存ペアを生成要求

        # 2つ目の数列と依存関係情報を生成
        dependent_sample = generate_dependent_sequence_sample(initial_sample, depth if depth else random.randint(3,8), is_x_bounded=is_bounded_request)

        if dependent_sample is not None:
            generated_data.append(dependent_sample)
            if dependent_sample['is_x_bounded'] == 1:
                current_dependent_count += 1
            else:
                current_independent_count += 1
            print(f"[DEBUG] Generated {len(generated_data)}/{num_samples} samples (Dep: {current_dependent_count}, Indep: {current_independent_count}) (Total attempts: {sample_generation_attempts})")
        else:
            # print(f"DEBUG: Dependent sample generation failed. Retrying main loop. (Attempt {sample_generation_attempts})")
            pass # generate_dependent_sequence_sample内で再試行が行われる

    if len(generated_data) < num_samples:
        print(f"WARNING: Could not generate {num_samples} samples after {max_total_attempts} total attempts. Generated only {len(generated_data)} samples.")
    
    print(f"[DEBUG] Finished generate_classification_data. Total dependent sequences: {current_dependent_count}. Total samples: {len(generated_data)}")
    return generated_data

'''
data = generate_learning_data(3, 10)
for sample in data:
    if(len(sample['numeric_sequence_1']) != 20 or len(sample['numeric_sequence_2']) != 20):
        print('error')

print(data)
print('finish')
'''