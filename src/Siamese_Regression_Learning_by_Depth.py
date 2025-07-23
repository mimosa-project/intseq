import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# プロジェクトの他のモジュールから関数をインポート
import program
import generate_program
import generate_data_set
import weight 

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 64
EPOCHS = 20 # 回帰は分類より収束に時間がかかることがあるため、多めに設定
BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.8 # 訓練データの割合 (例: 0.8なら訓練80%, テスト20%)

# --- モデル定義の関数 (Siamese_Regression_Learning.py から再利用) ---
def create_encoder(input_shape):
    input_seq = keras.Input(shape=input_shape, name="input_sequence")
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_seq)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(EMBEDDING_DIM, activation='relu')(x)
    return keras.Model(inputs=input_seq, outputs=x, name="encoder")

def create_siamese_network_for_regression(input_shape):
    encoder = create_encoder(input_shape)

    input_A = keras.Input(shape=input_shape, name="input_A")
    input_B = keras.Input(shape=input_shape, name="input_B")
    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)

    # 埋め込みベクトルを結合
    merged_features = layers.concatenate([embedding_A, embedding_B], axis=-1)

    # 結合された特徴量から最終的な出力を予測する層
    x = layers.Dense(64, activation='relu')(merged_features)
    x = layers.Dropout(0.3)(x) 
    x = layers.Dense(32, activation='relu')(x)

    # 回帰のための出力層: 連続値（情報量）を予測
    output_layer = layers.Dense(1, activation='linear', name='predicted_information_amount')(x)
    loss_function = 'mse' 
    metrics = ['mae'] 

    siamese_model = keras.Model(inputs=[input_A, input_B], outputs=output_layer, name="siamese_information_amount_model")

    return siamese_model, encoder, loss_function, metrics

# 対数変換を適用する関数
def log_transform_with_sign(seq):
    seq_np = np.array(seq, dtype=np.float64)
    transformed_seq = np.sign(seq_np) * np.log1p(np.abs(seq_np))
    return transformed_seq.tolist()


# --- メイン処理 ---
if __name__ == "__main__":
    input_shape = (MAX_SEQUENCE_LENGTH, 1)
    
    # ★ 1. ロードするデータセットファイルの指定 ★
    # ここに読み込みたい深さ別のデータセットファイル (.pkl) をリストで指定してください。
    # 例: 特定の深さのファイルを指定する場合
    # DATASET_FILES = ["generated_datasets/classification_data_fixed_depth_7.pkl"]
    # 例: 複数の深さのファイルを結合して使用する場合
    DATASET_FILES = [
        "generated_datasets/classification_data_fixed_depth_1.pkl",
        "generated_datasets/classification_data_fixed_depth_2.pkl",
        "generated_datasets/classification_data_fixed_depth_3.pkl",
        "generated_datasets/classification_data_fixed_depth_4.pkl",
        "generated_datasets/classification_data_fixed_depth_5.pkl",
        "generated_datasets/classification_data_fixed_depth_6.pkl",
        "generated_datasets/classification_data_fixed_depth_7.pkl",
        "generated_datasets/classification_data_fixed_depth_8.pkl",
        "generated_datasets/classification_data_fixed_depth_9.pkl",
        "generated_datasets/classification_data_fixed_depth_10.pkl"
    ]

    # ★ 訓練・評価に使用するサンプル数を指定 ★
    # ロードされた全データから、この数だけランダムにサンプリングして使用します。
    # メモリ不足を避けるため、または計算時間を抑えるために設定してください。
    NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION = 400000 


    # --- 2. データセットのロードと結合 ---
    all_raw_samples = [] 

    print("\n--- Loading datasets from specified files ---")
    for file_path in DATASET_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue
        
        start_time_load = time.time()
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            end_time_load = time.time()
            print(f"Loaded {len(data)} samples from {file_path} in {end_time_load - start_time_load:.2f} seconds.")
            all_raw_samples.extend(data) # リストに追加
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this file.")
            continue
    
    if not all_raw_samples:
        print("Error: No data loaded from any specified files. Please check DATASET_FILES paths.")
        exit()

    print(f"\nTotal raw samples loaded: {len(all_raw_samples)}")

    # --- 3. 必要サンプル数のチェックとサンプリング ---
    if len(all_raw_samples) < NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"ERROR: Total loaded samples ({len(all_raw_samples)}) is less than required ({NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Aborting.")
        exit()
    elif len(all_raw_samples) > NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"Total loaded samples ({len(all_raw_samples)}) exceeds required ({NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Sampling randomly...")
        
        indices = np.arange(len(all_raw_samples))
        np.random.shuffle(indices)
        sampled_indices = indices[:NUM_SAMPLES_FOR_TRAINING_AND_EVALUATION]
        sampled_samples_raw = [all_raw_samples[i] for i in sampled_indices]
        
        print(f"Sampled down to {len(sampled_samples_raw)} samples for training and evaluation.")
    else:
        sampled_samples_raw = all_raw_samples
        print(f"Using all {len(sampled_samples_raw)} loaded samples for training and evaluation.")
    
    # --- 4. 'is_x_bounded' が 1 のサンプルのみをフィルタリング ---
    dependent_data = [d for d in sampled_samples_raw if d['is_x_bounded'] == 1]
    
    if not dependent_data:
        print("ERROR: No dependent samples found after filtering. Cannot train regression model. Aborting.")
        exit()

    print(f"Filtered {len(dependent_data)} dependent samples for regression training.")

    # 訓練・評価用のデータ準備
    X_A_raw = [s['numeric_sequence_1'] for s in dependent_data]
    X_B_raw = [s['numeric_sequence_2'] for s in dependent_data]
    y_info_amount_raw = [s['transformed_sequence_info_amount'] for s in dependent_data] 

    # スケーリングとNumPy変換
    X_A_scaled = np.array([log_transform_with_sign(seq) for seq in X_A_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, 1)
    X_B_scaled = np.array([log_transform_with_sign(seq) for seq in X_B_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, 1)
    y_info_amount = np.array(y_info_amount_raw, dtype=np.float32)

    # 訓練データとテストデータに分割
    X_train_A, X_test_A, X_train_B, X_test_B, \
    y_train, y_test = train_test_split(
        X_A_scaled, X_B_scaled, y_info_amount,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42
    )
    
    print("\nTraining and Test data shapes:")
    print(f"X_train_A shape: {X_train_A.shape}")
    print(f"X_train_B shape: {X_train_B.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test_A shape: {X_test_A.shape}")
    print(f"X_test_B shape: {X_test_B.shape}")
    print(f"y_test shape: {y_test.shape}")


    # --- 5. モデルの作成と訓練 ---
    siamese_reg_model, encoder_reg, loss_reg, metrics_reg = create_siamese_network_for_regression(input_shape)
    siamese_reg_model.compile(optimizer='adam', loss=loss_reg, metrics=metrics_reg)

    print("\n--- Starting regression model training ---")
    history = siamese_reg_model.fit(
        [X_train_A, X_train_B],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # 訓練データの一部を検証データとして使用
        verbose=1
    )
    print("Regression model training finished.")

    # --- 6. モデルの評価 ---
    print("\n--- Evaluating on Test Set (Regression) ---")
    loss, mae = siamese_reg_model.evaluate([X_test_A, X_test_B], y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    y_pred = siamese_reg_model.predict([X_test_A, X_test_B]).flatten()

    test_mse_sklearn = mean_squared_error(y_test, y_pred)
    test_mae_sklearn = mean_absolute_error(y_test, y_pred) # MAEはKerasのmetricsにもあるが、sklearnでも確認
    test_r2 = r2_score(y_test, y_pred)

    print(f"Test MSE (sklearn): {test_mse_sklearn:.4f}")
    print(f"Test MAE (sklearn): {test_mae_sklearn:.4f}")
    print(f"Test R^2 Score: {test_r2:.4f}")

    print("\n--- Analysis finished ---")