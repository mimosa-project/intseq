import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import time

# プロジェクトの他のモジュールから関数をインポート
import program
import generate_program
import generate_data_set
import weight 

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 64
EPOCHS = 20 
BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.8 

# --- 分析に使用するサンプル数 ---
MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION = 100000 


# ★ 編集距離回帰に適した距離指標を計算するカスタムレイヤー ★
class EditDistanceFriendlyFeatures(layers.Layer):
    def __init__(self, **kwargs):
        super(EditDistanceFriendlyFeatures, self).__init__(**kwargs)
    
    def call(self, inputs):
        embedding_A, embedding_B = inputs
        
        # 1. ユークリッド距離（編集距離と最も相関が高い）
        euclidean_dist = tf.sqrt(tf.reduce_sum(tf.square(embedding_A - embedding_B), axis=-1, keepdims=True))
        
        # 2. マンハッタン距離（L1距離）
        manhattan_dist = tf.reduce_sum(tf.abs(embedding_A - embedding_B), axis=-1, keepdims=True)
        
        # 3. チェビシェフ距離（L∞距離）
        chebyshev_dist = tf.reduce_max(tf.abs(embedding_A - embedding_B), axis=-1, keepdims=True)
        
        # 4. 要素ごとの差の二乗和（ユークリッド距離の二乗）
        squared_diff_sum = tf.reduce_sum(tf.square(embedding_A - embedding_B), axis=-1, keepdims=True)
        
        # 5. コサイン距離（1 - コサイン類似度）を編集距離らしくスケール
        embedding_A_norm = tf.nn.l2_normalize(embedding_A, axis=-1)
        embedding_B_norm = tf.nn.l2_normalize(embedding_B, axis=-1)
        cosine_sim = tf.reduce_sum(embedding_A_norm * embedding_B_norm, axis=-1, keepdims=True)
        cosine_distance = 1.0 - cosine_sim  # 0〜2の範囲
        
        # すべての距離指標を結合
        distance_features = tf.concat([
            euclidean_dist, 
            manhattan_dist, 
            chebyshev_dist, 
            squared_diff_sum, 
            cosine_distance
        ], axis=-1)
        
        return distance_features

# ★ より編集距離に近い形の活性化関数を持つカスタムレイヤー ★
class EditDistanceActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(EditDistanceActivation, self).__init__(**kwargs)
    
    def call(self, inputs):
        # ReLUベースだが、編集距離らしい非線形性を持たせる
        # 小さい値では線形に近く、大きい値では増加率が鈍化
        # tf.nn.relu(inputs) + 1e-7 は、入力が負にならないようにし、平方根計算の安定性を確保します。
        return tf.sqrt(tf.nn.relu(inputs) + 1e-7)  # 平方根で非線形性を調整

# ★ 編集距離予測に最適化されたエンコーダー ★
def create_encoder_for_edit_distance(input_shape, embedding_dim=64):
    """編集距離予測に最適化されたエンコーダー"""
    input_seq = keras.Input(shape=input_shape, name="input_sequence")
        
    # 位置情報を重視するためのConv1D
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_seq)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
        
    # 順序情報を保持するLSTM
    x = layers.LSTM(64, return_sequences=False)(x)
        
    # 編集距離計算に有効な特徴量を抽出
    x = layers.Dense(embedding_dim, activation='relu')(x)

    # バッチ正規化
    x = layers.BatchNormalization()(x)
        
    # L2正規化で埋め込み空間を安定化
    x = tf.nn.l2_normalize(x, axis=-1)
        
    return keras.Model(inputs=input_seq, outputs=x, name="edit_distance_encoder")

# ★ 編集距離回帰に特化したSiameseネットワーク ★
def create_edit_distance_siamese_network(input_shape, embedding_dim=64, max_edit_distance=None):
    """編集距離回帰に特化したSiameseネットワーク"""
    encoder = create_encoder_for_edit_distance(input_shape, embedding_dim)
    input_A = keras.Input(shape=input_shape, name="input_A")
    input_B = keras.Input(shape=input_shape, name="input_B")
    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)
    
    # 編集距離に適した距離特徴量を計算
    distance_features = EditDistanceFriendlyFeatures()([embedding_A, embedding_B])
        
    # 距離特徴量を編集距離に変換
    x = layers.Dense(8, activation='relu')(distance_features)

    # バッチ正規化
    x = layers.BatchNormalization()(x)

    x = EditDistanceActivation()(x) # カスタム活性化関数
        
    # 最終的な編集距離予測
    output_layer = layers.Dense(1, activation='relu', name='predicted_edit_distance')(x)
        
    # 最大編集距離でクリッピング（オプション）
    if max_edit_distance is not None:
        output_layer = tf.clip_by_value(output_layer, 0.0, max_edit_distance)
    
    siamese_model = keras.Model(inputs=[input_A, input_B], outputs=output_layer, 
                               name="edit_distance_siamese_model")
        
    return siamese_model, encoder

# ★ 最もシンプルな編集距離近似（ユークリッド距離のみ） ★
def create_simple_euclidean_siamese_network(input_shape, embedding_dim=64):
    """最もシンプルな編集距離近似（ユークリッド距離のみ）"""
    encoder = create_encoder_for_edit_distance(input_shape, embedding_dim)
    input_A = keras.Input(shape=input_shape, name="input_A")
    input_B = keras.Input(shape=input_shape, name="input_B")
    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)
    
    # ユークリッド距離を直接計算
    euclidean_dist = tf.sqrt(tf.reduce_sum(tf.square(embedding_A - embedding_B), axis=-1, keepdims=True))
        
    # 学習可能なスケーリング
    output_layer = layers.Dense(1, activation='relu', use_bias=True, 
                               name='predicted_edit_distance')(euclidean_dist)
    
    # バッチ正規化
    output_layer = layers.BatchNormalization()(output_layer)
    
    siamese_model = keras.Model(inputs=[input_A, input_B], outputs=output_layer, 
                               name="euclidean_siamese_model")
        
    return siamese_model, encoder

# ★ Contrastive Lossを使用した編集距離学習（ただし、回帰タスクには要調整） ★
def create_contrastive_loss_siamese_network(input_shape, embedding_dim=64, margin=2.0):
    """Contrastive Lossを使用した編集距離学習"""
    # このモデルはユークリッド距離を出力し、Contrastive Lossで学習することを想定
    # 回帰タスクの損失関数 (MSE/MAE) を適用するには、この出力がそのまま編集距離になるよう調整が必要
    encoder = create_encoder_for_edit_distance(input_shape, embedding_dim)
    input_A = keras.Input(shape=input_shape, name="input_A")
    input_B = keras.Input(shape=input_shape, name="input_B")
    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)
    
    # ユークリッド距離
    euclidean_dist = tf.sqrt(tf.reduce_sum(tf.square(embedding_A - embedding_B), axis=-1, keepdims=True))
    
    siamese_model = keras.Model(inputs=[input_A, input_B], outputs=euclidean_dist, 
                               name="contrastive_siamese_model")
        
    return siamese_model, encoder

# ★ カスタム損失関数：編集距離に最適化 ★
def edit_distance_loss(y_true, y_pred):
    """編集距離予測に最適化された損失関数"""
    # 基本のMSE
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
    # 小さい編集距離での精度を重視する重み
    # y_trueが0の場合にinfにならないよう、y_true + 1.0
    weights = 1.0 / (y_true + 1.0)  
    weighted_mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        
    return 0.7 * mse_loss + 0.3 * weighted_mse # 重み付けの割合は調整可能

# ★ 編集距離回帰に推奨されるモデル構成を選択・コンパイルするヘルパー関数 ★
def get_recommended_model_for_edit_distance(input_shape, embedding_dim=64, max_edit_distance=None):
    """編集距離回帰に推奨されるモデル構成"""
        
    # ここで試したいモデルを選択します。
    # model, encoder = create_simple_euclidean_siamese_network(input_shape, embedding_dim) # 1. 最もシンプル
    model, encoder = create_edit_distance_siamese_network(input_shape, embedding_dim, max_edit_distance) # 2. より高機能な版 (推奨)
    # model, encoder = create_contrastive_loss_siamese_network(input_shape, embedding_dim, max_edit_distance) # 3. Contrastive Loss用 (損失関数もContrastiveLossにする必要あり)
        
    # 編集距離に最適化された損失関数とメトリクス
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=edit_distance_loss,  # ★ カスタム損失関数を適用 ★
        metrics=['mae', 'mse']    # MAEとMSEを評価指標に
    )
        
    return model, encoder

# 対数変換を適用する関数
def log_transform_with_sign(seq):
    seq_np = np.array(seq, dtype=np.float64)
    # np.log1p は log(1+x) を計算し、xが0に近い場合でも安定
    # np.abs は絶対値、np.sign は符号（正なら1, 負なら-1, 0なら0）
    transformed_seq = np.sign(seq_np) * np.log1p(np.abs(seq_np))
    return transformed_seq.tolist()

# --- メイン処理 ---
if __name__ == "__main__":
    input_shape = (MAX_SEQUENCE_LENGTH, 1)
    
    # ★ 1. ロードするデータセットファイルの指定 ★
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
            all_raw_samples.extend(data) 
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
    # ここでの 'is_x_bounded' は、データセットから読み込んだそのままの値
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

    # NaN/Inf チェックの追加
    if np.any(np.isnan(X_A_scaled)): print("DEBUG: NaN found in X_A_scaled!")
    if np.any(np.isinf(X_A_scaled)): print("DEBUG: Inf found in X_A_scaled!")
    if np.any(np.isnan(X_B_scaled)): print("DEBUG: Inf found in X_B_scaled!")
    if np.any(np.isinf(X_B_scaled)): print("DEBUG: Inf found in X_B_scaled!")
    if np.any(np.isnan(y_info_amount)): print("DEBUG: NaN found in y_info_amount!")
    if np.any(np.isinf(y_info_amount)): print("DEBUG: Inf found in y_info_amount!")

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
    # get_recommended_model_for_edit_distance を呼び出してモデルとエンコーダを取得
    # この関数内でモデルのコンパイルも行われるため、別途 compile は不要
    siamese_reg_model, encoder_reg = get_recommended_model_for_edit_distance(input_shape, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    
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

    loss_val, mae_val, mse_val = siamese_reg_model.evaluate([X_test_A, X_test_B], y_test, verbose=0) # ★ ここを修正 ★
    
    print(f"Test Loss (edit_distance_loss): {loss_val:.4f}") 
    print(f"Test MAE (from Keras): {mae_val:.4f}")          
    print(f"Test MSE (from Keras): {mse_val:.4f}")          

    y_pred = siamese_reg_model.predict([X_test_A, X_test_B]).flatten()

    # 予測結果の NaN/Inf チェックの追加
    if np.any(np.isnan(y_pred)): print("DEBUG: NaN found in y_pred!")
    if np.any(np.isinf(y_pred)): print("DEBUG: Inf found in y_pred!")

    test_mse_sklearn = mean_squared_error(y_test, y_pred)
    test_mae_sklearn = mean_absolute_error(y_test, y_pred) 
    test_r2 = r2_score(y_test, y_pred)

    print(f"Test MSE (sklearn): {test_mse_sklearn:.4f}")
    print(f"Test MAE (sklearn): {test_mae_sklearn:.4f}")
    print(f"Test R^2 Score: {test_r2:.4f}")

    print("\n--- Analysis finished ---")