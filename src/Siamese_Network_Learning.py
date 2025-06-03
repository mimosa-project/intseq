import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import program
import generate_data_set
from sklearn.model_selection import train_test_split # sklearnを使ってデータを分割
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score # 評価指標

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

    # モデルの概要を表示
    siamese_dep_model.summary()

    # モデルのコンパイル
    siamese_dep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- 訓練データとテストデータの生成と分割 ---
    num_total_samples = 10000 
    # 全データを生成
    raw_data = generate_data_set.generate_classification_data(num_samples=num_total_samples)

    # データ整形 (Numpy配列に変換し、必要な次元を追加)
    X_A_raw = np.array([d for d in raw_data['numeric_sequence_1']]).reshape(num_total_samples, MAX_SEQUENCE_LENGTH, 1)
    X_B_raw = np.array([d for d in raw_data['numeric_sequence_2']]).reshape(num_total_samples, MAX_SEQUENCE_LENGTH, 1)
    y_raw = np.array(raw_data['is_x_bounded'])

    # 訓練データとテストデータに分割 (例: 訓練80%, テスト20%)
    # `stratify=y_raw` を指定することで、クラスの割合を訓練セットとテストセットで維持します
    X_train_A, X_test_A, X_train_B, X_test_B, y_train, y_test = train_test_split(
        X_A_raw, X_B_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # 訓練データの形状を表示
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
        validation_split=0.2 # 訓練データの一部を検証データとして使用
    )
    print("Training finished.")

    # --- テストデータでの最終評価 ---
    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = siamese_dep_model.evaluate([X_test_A, X_test_B], y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # その他の評価指標
    y_pred_prob = siamese_dep_model.predict([X_test_A, X_test_B])
    y_pred = (y_pred_prob > 0.5).astype(int) # 閾値0.5で二値化

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")

    # --- (オプション) モデルの保存 ---
    # siamese_dep_model.save("siamese_dependency_model.h5")
    # encoder_dep.save("dependency_encoder.h5")