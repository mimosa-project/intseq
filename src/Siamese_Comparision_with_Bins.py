# 分位ビニング追加前
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

# プロジェクトの他のモジュールから関数をインポート
import program
import generate_program
import generate_data_set
import weight

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 64
N_BINS = 20
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_TEST_SPLIT_RATIO = 0.8 

# --- 分析に使用するサンプル数 ---
MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION = 100000 


# --- 分位ビン化用関数 ---
def compute_bin_edges(all_sequences, n_bins=20):
    # 全数列を1Dに平坦化し，ビン範囲を決定
    # 出力は境界点のリスト n_bins+1
    all_values = np.concatenate(all_sequences)
    edges = np.quantile(all_values, np.linspace(0, 1, n_bins+1))
    return edges

def sequence_to_bins(seq, edges):
    bins = np.searchsorted(edges, seq, side='right') - 1
    return np.clip(bins, 0, len(edges)-2)  # IDは0〜n_bins-1


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

def create_encoder_for_edit_distance_with_bins(raw_input_shape, bins_input_shape, n_bins, embedding_dim=64):
    """分位ビニングを追加したエンコーダー"""

    # --元のfloat入力部分--    
    input_raw = keras.Input(shape=raw_input_shape, name="raw_sequence")

    # 位置情報を重視するためのConv1D
    x_raw = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_raw)
    x_raw = layers.Dropout(0.5)(x_raw)
    x_raw = layers.MaxPooling1D(pool_size=2)(x_raw)
    x_raw = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x_raw)
    x_raw = layers.Dropout(0.5)(x_raw)
    x_raw = layers.MaxPooling1D(pool_size=2)(x_raw)
        
    # 順序情報を保持するLSTM
    x_raw = layers.LSTM(64, return_sequences=False)(x_raw)
    x_raw = layers.Dropout(0.5)(x_raw)
          

    # --ビン入力部分--
    # 生データを同じ時系列データのため，同じ処理を通す
    input_bins = keras.Input(shape=bins_input_shape, dtype='int32')
    x_bins = layers.Embedding(input_dim=n_bins+1, output_dim=16)(input_bins)
    x_bins = layers.Conv1D(32, 3, activation='relu', padding='same')(x_bins)
    x_bins = layers.Dropout(0.5)(x_bins)
    x_bins = layers.MaxPooling1D(2)(x_bins)
    x_bins = layers.Conv1D(64, 3, activation='relu', padding='same')(x_bins)
    x_bins = layers.Dropout(0.5)(x_bins)
    x_bins = layers.MaxPooling1D(2)(x_bins)
    x_bins = layers.LSTM(64)(x_bins) 
    x_bins = layers.Dropout(0.5)(x_bins)


    # --結合--
    x = layers.concatenate([x_raw, x_bins])

    # 編集距離計算に有効な特徴量を抽出(ビンデータ分の情報が増えるため次元数を上げた方が良い？この場合過学習対策推奨)
    x = layers.Dense(embedding_dim, activation='relu')(x)

    # バッチ正規化
    x = layers.BatchNormalization()(x)

    # Dropout
    x = layers.Dropout(0.3)(x)
        
    # L2正規化で埋め込み空間を安定化
    x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)
        
    return keras.Model(inputs=[input_raw, input_bins], outputs=x, name="edit_distance_encoder_with_bins")

# 対数変換を適用する関数
def log_transform_with_sign(seq):
    seq_np = np.array(seq, dtype=np.float64)
    # np.log1p は log(1+x) を計算し、xが0に近い場合でも安定
    # np.abs は絶対値、np.sign は符号（正なら1, 負なら-1, 0なら0）
    transformed_seq = np.sign(seq_np) * np.log1p(np.abs(seq_np))
    return transformed_seq.tolist()

def create_comparison_classifier_model_with_bins(raw_input_shape, bin_input_shape, n_bins, embedding_dim=EMBEDDING_DIM):
    # エンコーダは共通で利用
    encoder = create_encoder_for_edit_distance_with_bins(
        raw_input_shape, 
        bins_input_shape, 
        n_bins, 
        embedding_dim)

    # 3組の入力: 数列A, 数列B, 数列C，それぞれのビン化
    inputA_raw = keras.Input(shape=raw_input_shape, name="input_A_comp")
    inputB_raw = keras.Input(shape=raw_input_shape, name="input_B_comp")
    inputC_raw = keras.Input(shape=raw_input_shape, name="input_C_comp")

    inputA_bins = keras.Input(shape=bin_input_shape, dtype='int32', name="A_bins")
    inputB_bins = keras.Input(shape=bin_input_shape, dtype='int32', name="B_bins")
    inputC_bins = keras.Input(shape=bin_input_shape, dtype='int32', name="C_bins")



    # 各数列を埋め込みベクトルに変換
    embedding_A = encoder([inputA_raw, inputA_bins])
    embedding_B = encoder([inputB_raw, inputB_bins])
    embedding_C = encoder([inputC_raw, inputC_bins])

    # AとBのペアの距離特徴量
    features_AB = EditDistanceFriendlyFeatures()([embedding_A, embedding_B])
    # AとCのペアの距離特徴量
    features_AC = EditDistanceFriendlyFeatures()([embedding_A, embedding_C])

    # 距離特徴量を結合し、比較のための分類ヘッドへ入力
    # 選択肢1: 単純結合（現状）
    # comparison_features = layers.concatenate([features_AB, features_AC], axis=-1)

    # 選択肢2: 差の絶対値のみ (より直接的)
    # comparison_features = tf.abs(features_AB - features_AC)

    # 選択肢3: 結合 + 差の絶対値 (最も推奨)
    diff_features = tf.abs(features_AB - features_AC)
    comparison_features = layers.concatenate([features_AB, features_AC, diff_features], axis=-1) 

    # 選択肢4: 結合 + 比率 (ゼロ除算注意)
    # ratio_features = tf.math.divide_no_nan(features_AB, features_AC)
    # comparison_features = layers.concatenate([features_AB, features_AC, ratio_features], axis=-1)


    # 分類ヘッド
    x = layers.Dense(64, activation='relu')(comparison_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.01)
                     )(x)
    x = layers.BatchNormalization()(x)
    
    # 最終的な出力層: 二値分類のためsigmoid活性化関数を使用
    output_layer = layers.Dense(1, activation='sigmoid', name='edit_distance_comparison_label')(x)

    # モデルの構築
    model = keras.Model(
        inputs=[inputA_raw, inputA_bins, inputB_raw, inputB_bins, inputC_raw, inputC_bins], 
        outputs=output_layer, 
        name="siamese_comparison_classifier"
    )

    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy', # 二値分類用損失関数
        metrics=['accuracy', 'Precision', 'Recall', 'AUC'] # 分類評価指標
    )
    
    return model, encoder

# --- メイン処理 ---
if __name__ == "__main__":
    raw_input_shape = (MAX_SEQUENCE_LENGTH, 1)
    bins_input_shape = (MAX_SEQUENCE_LENGTH,)
    
    # ★ 1. ロードするデータセットファイルの指定 ★
    DATASET_FILES = [
        "comparison_datasets/comparison_data_depth_1-10.pkl"# generate_comparison_datasets.py で生成したファイル名に合わせる
        # 例: 複数の深さの比較データを結合する場合
        # "comparison_datasets/comparison_data_depth_1-10.pkl",
        # "comparison_datasets/comparison_data_depth_11-15.pkl",
    ]

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
    if len(all_raw_samples) < MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"ERROR: Total loaded samples ({len(all_raw_samples)}) is less than required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Aborting.")
        exit()
    elif len(all_raw_samples) > MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION:
        print(f"Total loaded samples ({len(all_raw_samples)}) exceeds required ({MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION}). Sampling randomly...")
        
        indices = np.arange(len(all_raw_samples))
        np.random.shuffle(indices)
        sampled_indices = indices[:MAX_SAMPLES_FOR_TRAINING_AND_EVALUATION]
        sampled_samples_raw = [all_raw_samples[i] for i in sampled_indices]
        
        print(f"Sampled down to {len(sampled_samples_raw)} samples for training and evaluation.")
    else:
        sampled_samples_raw = all_raw_samples
        print(f"Using all {len(sampled_samples_raw)} loaded samples for training and evaluation.")
    
    # --- 4. データ準備 (3つの入力) ---
    X_A_raw = [s['numeric_sequence_A'] for s in sampled_samples_raw]
    X_B_raw = [s['numeric_sequence_B'] for s in sampled_samples_raw]
    X_C_raw = [s['numeric_sequence_C'] for s in sampled_samples_raw]
    y_labels_raw = [s['target_label'] for s in sampled_samples_raw] 

    # スケーリングとNumPy変換
    # スケーリングのタイミングに注意した方が良いか，ビン化後にスケーリングする方がいい？
    X_A_scaled = np.array([log_transform_with_sign(seq) for seq in X_A_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, 1)
    X_B_scaled = np.array([log_transform_with_sign(seq) for seq in X_B_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, 1)
    X_C_scaled = np.array([log_transform_with_sign(seq) for seq in X_C_raw], dtype=np.float32).reshape(-1, MAX_SEQUENCE_LENGTH, 1)
    y_labels = np.array(y_labels_raw, dtype=np.int32)

    # NaN/Inf チェックの追加
    if np.any(np.isnan(X_A_scaled)): print("DEBUG: NaN found in X_A_scaled!")
    if np.any(np.isinf(X_A_scaled)): print("DEBUG: Inf found in X_A_scaled!")
    if np.any(np.isnan(X_B_scaled)): print("DEBUG: NaN found in X_B_scaled!")
    if np.any(np.isinf(X_B_scaled)): print("DEBUG: Inf found in X_B_scaled!")
    if np.any(np.isnan(X_C_scaled)): print("DEBUG: NaN found in X_C_scaled!")
    if np.any(np.isinf(X_C_scaled)): print("DEBUG: Inf found in X_C_scaled!")
    if np.any(np.isnan(y_labels)): print("DEBUG: NaN found in y_labels!")
    if np.any(np.isinf(y_labels)): print("DEBUG: Inf found in y_labels!")

    # 訓練データとテストデータに分割
    # 3つの入力 (X_A, X_B, X_C) と1つのターゲット (y_labels)
    X_train_A_raw, X_test_A_raw, X_train_B_raw, X_test_B_raw, X_train_C_raw, X_test_C_raw, \
    y_train, y_test = train_test_split(
        X_A_scaled, X_B_scaled, X_C_scaled, y_labels,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO, 
        random_state=42,
        stratify=y_labels # 分類問題なのでstratifyでラベル分布を維持
    )

    # --- 5. 訓練データを用いてビン化 ---

    # 訓練データから分位境界を求める
    edges = compute_bin_edges(list(X_train_A_raw) + list(X_train_B_raw) + list(X_train_C_raw), 
                              n_bins=N_BINS)

    # 訓練データの分位境界を用いて，データを分位ビニング
    X_train_A_bins = np.array([sequence_to_bins(seq, edges) for seq in X_train_A_raw])
    X_train_B_bins = np.array([sequence_to_bins(seq, edges) for seq in X_train_B_raw])
    X_train_C_bins = np.array([sequence_to_bins(seq, edges) for seq in X_train_C_raw])

    X_test_A_bins = np.array([sequence_to_bins(seq, edges) for seq in X_test_A_raw])
    X_test_B_bins = np.array([sequence_to_bins(seq, edges) for seq in X_test_B_raw])
    X_test_C_bins = np.array([sequence_to_bins(seq, edges) for seq in X_test_C_raw])

    # データ確認
    print("\nTraining and Test data shapes:")
    print(f"X_train_A_raw shape: {X_train_A_raw.shape}")
    print(f"X_train_B_raw shape: {X_train_B_raw.shape}")
    print(f"X_train_C_raw shape: {X_train_C_raw.shape}")
    print(f"X_train_A_bins shape: {X_train_A_bins.shape}")
    print(f"X_train_B_bins shape: {X_train_B_bins.shape}")
    print(f"X_train_C_bins shape: {X_train_C_bins.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test_A_raw shape: {X_test_A_raw.shape}")
    print(f"X_test_B_raw shape: {X_test_B_raw.shape}")
    print(f"X_test_C_raw shape: {X_test_C_raw.shape}")
    print(f"X_test_A_bins shape: {X_test_A_bins.shape}")
    print(f"X_test_B_bins shape: {X_test_B_bins.shape}")
    print(f"X_test_C_bins shape: {X_test_C_bins.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 5. モデルの作成と訓練 ---
    model, encoder = create_comparison_classifier_model_with_bins(raw_input_shape, bins_input_shape, N_BINS, EMBEDDING_DIM) # 新しいモデルを呼び出し
    
    print("\n--- Starting comparison model training ---")
    history = model.fit(
        [X_train_A_raw, X_train_A_bins, X_train_B_raw, X_train_B_bins, X_train_C_raw, X_train_C_bins], # 3つの入力
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # 訓練データの一部を検証データとして使用
        verbose=1
    )
    print("Comparison model training finished.")

    # --- 6. モデルの評価 ---
    print("\n--- Evaluating on Test Set (Comparison) ---")
    # model.evaluate は compile 時に指定された metrics を返す
    # accuracy, precision, recall, auc の順で受け取る
    loss, accuracy, precision, recall, auc = model.evaluate([X_test_A_raw, X_test_A_bins, X_test_B_raw, X_test_B_bins, X_test_C_raw, X_test_C_bins], y_test, verbose=0)
    
    # 予測の実行 (混同行列やF1スコア計算のため)
    y_pred_prob = model.predict([X_test_A_raw, X_test_A_bins, X_test_B_raw, X_test_B_bins, X_test_C_raw, X_test_C_bins]).flatten()
    y_pred_binary = (y_pred_prob > 0.5).astype(int) # 0.5 を閾値として二値化

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score(y_test, y_pred_binary):.4f}") # scikit-learnのf1_scoreを使用
    print(f"Test AUC-ROC: {auc:.4f}")

    # 混同行列も表示 (オプション)
    # cm = confusion_matrix(y_test, y_pred_binary)
    # print("\nConfusion Matrix:")
    # print(cm)

    print("\n--- Analysis finished ---")