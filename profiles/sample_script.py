import sys
sys.path.append('../intseq')
from src import program, sequence_generator

if __name__ == "__main__":
    import cProfile

    #プロファイリングを開始
    profiler = cProfile.Profile()
    profiler.enable()

    # プロファイリング対象
    sequence_list=[]
    for num in range(100):
        while(True):
            try:
                sequence=[]
                seq, info = sequence_generator.select_sequence(100, 10)
                ps=program.ProgramStack(program.ProgramStack.str2rpn(seq), 10)
                stack=ps.build()
                for i in range(10):
                    sequence.append(stack[0].calc(i, 0))
                sequence_list.append(sequence)
                break

            except Exception:
                continue
    
    # プロファイルリングを停止
    profiler.disable()

    # 保存
    profiler.dump_stats('profiles/profile_data_after_correcting_generate_random_letter.prof')

    # 数列サンプルを保存
    with open('profiles/after_correcting_generate_random_letter.txt', 'w') as file:
        for num in range(100):
            file.write(f"{sequence_list[num]}\n")
