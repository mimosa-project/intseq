import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import program
import generate_data_set
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import cProfile # cProfileをインポート
import pstats   # pstatsをインポート（結果をファイルに保存するため）

# --- ハイパーパラメータの定義 ---
MAX_SEQUENCE_LENGTH = 20  # 入力要素数20の数列
EMBEDDING_DIM = 64         # 埋め込みベクトルの次元数

# --- エンコーダ（共有重み）の定義 ---
def create_encoder(input_shape):
    input_seq = keras.Input(shape=input_shape, name="input_sequence")
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_seq)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(EMBEDDING_DIM, activation='relu')(x)
    return keras.Model(inputs=input_seq, outputs=x, name="encoder")

# --- Siamese Networkの構築（依存関係分類用） ---
def create_siamese_network_for_dependency(input_shape):
    encoder = create_encoder(input_shape)

    input_A = keras.Input(shape=input_shape, name="input_A")
    input_B = keras.Input(shape=input_shape, name="input_B")

    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)

    # 埋め込みベクトルの差の絶対値を特徴量として利用
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_A, embedding_B])

    # 結合層と出力層
    merged = layers.Dense(64, activation='relu')(distance)
    merged = layers.Dropout(0.1)(merged)
    output_layer = layers.Dense(1, activation='sigmoid', name="dependency_probability")(merged)

    siamese_model = keras.Model(inputs=[input_A, input_B], outputs=output_layer, name="siamese_dependency_model")
    return siamese_model, encoder

# --- メイン処理（モデルの構築と訓練） ---
if __name__ == "__main__":
    input_shape = (MAX_SEQUENCE_LENGTH, 1)
    siamese_dep_model, encoder_dep = create_siamese_network_for_dependency(input_shape)

    siamese_dep_model.summary()
    siamese_dep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- 訓練データとテストデータの生成と分割 ---
    num_total_samples = 10000 

    print(f"Generating {num_total_samples} samples for profiling...")

    # プロファイリングを開始
    profiler = cProfile.Profile()
    profiler.enable()

    raw_data_list = generate_data_set.generate_classification_data(num_samples=num_total_samples)

    profiler.disable() # プロファイリングを終了

    # プロファイリング結果をファイルに保存
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative') # 累積時間でソート
    stats.dump_stats('data_generation_profile.prof') # 結果をファイルに保存

    print("Data generation profiling finished. Results saved to data_generation_profile.prof")


    # データをNumpy配列に変換し、スケーリングを行う
    numeric_sequence_1_list = [d['numeric_sequence_1'] for d in raw_data_list]
    numeric_sequence_2_list = [d['numeric_sequence_2'] for d in raw_data_list]
    is_x_bounded_list = [d['is_x_bounded'] for d in raw_data_list]

    def log_transform_with_sign(seq):
        seq_np = np.array(seq, dtype=np.float64)
        transformed_seq = np.sign(seq_np) * np.log1p(np.abs(seq_np))
        return transformed_seq.tolist()

    X_A_scaled = np.array([log_transform_with_sign(seq) for seq in numeric_sequence_1_list], dtype=np.float32).reshape(num_total_samples, MAX_SEQUENCE_LENGTH, 1)
    X_B_scaled = np.array([log_transform_with_sign(seq) for seq in numeric_sequence_2_list], dtype=np.float32).reshape(num_total_samples, MAX_SEQUENCE_LENGTH, 1)
    y = np.array(is_x_bounded_list, dtype=np.int32)

    if np.any(np.isnan(X_A_scaled)) or np.any(np.isinf(X_A_scaled)):
        print("Warning: NaN or Inf values found in X_A_scaled after log transformation. Review your data or transformation.")
    if np.any(np.isnan(X_B_scaled)) or np.any(np.isinf(X_B_scaled)):
        print("Warning: NaN or Inf values found in X_B_scaled after log transformation. Review your data or transformation.")

    X_train_A, X_test_A, X_train_B, X_test_B, y_train, y_test = train_test_split(
        X_A_scaled, X_B_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining and Test data shapes:")
    print(f"X_train_A shape: {X_train_A.shape}")
    print(f"X_train_B shape: {X_train_B.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test_A shape: {X_test_A.shape}")
    print(f"X_test_B shape: {X_test_B.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- モデルの訓練 ---
    epochs = 10
    batch_size = 32
    print("\nStarting model training...")
    history = siamese_dep_model.fit(
        [X_train_A, X_train_B],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    print("Training finished.")

    # --- テストデータでの最終評価 ---
    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = siamese_dep_model.evaluate([X_test_A, X_test_B], y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_prob = siamese_dep_model.predict([X_test_A, X_test_B])
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")

    # siamese_dep_model.save("siamese_dependency_model.h5")
    # encoder_dep.save("dependency_encoder.h5")