import tensorflow as tf
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 1. Data Loading
print("Starting Training with Best Hyperparameters...")

X_train = np.load('../results/X_train.npy')
X_test = np.load('../results/X_test.npy')
y_train = np.load('../results/y_train.npy')
y_test = np.load('../results/y_test.npy')
feature_names = np.load('../results/feature_names.npy', allow_pickle=True)

# 2. Mixed Precision Policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 3. Best Hyperparameters from Tuner
filters = 416
lstm_units = 32
dropout_rate = 0.2
learning_rate = 1e-4
num_heads = 10
key_dim = 32
kernel_size = 5

# 4. Model Definition (Using the best hyperparameters)
input_layer = layers.Input(shape=(X_train.shape[1], 1))

# CNN block
cnn_layer = layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          activation='relu',
                          padding='same')(input_layer)
cnn_layer = layers.BatchNormalization()(cnn_layer)
cnn_layer = layers.MaxPooling1D(pool_size=2)(cnn_layer)
cnn_output = layers.GlobalAveragePooling1D()(cnn_layer)

# LSTM block
lstm_input = layers.Reshape((1, -1))(cnn_output)
lstm_layer = layers.LSTM(units=lstm_units, return_sequences=True)(lstm_input)
lstm_layer = layers.LSTM(units=lstm_units * 2, return_sequences=False)(lstm_layer)

# Multi-Head Attention block
attention_output = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=key_dim,
    attention_axes=(1,)
)(lstm_layer, lstm_layer)
attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

# Dense + Dropout
dense_layer = layers.Dense(units=filters, activation='relu')(attention_output)
dense_layer = layers.Dropout(rate=dropout_rate)(dense_layer)

# Output
output_layer = layers.Dense(1, activation='sigmoid')(dense_layer)

# Compile model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Training
early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train.reshape(-1, X_train.shape[1], 1),
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# 6. Evaluation
y_pred_prob = model.predict(X_test.reshape(-1, X_test.shape[1], 1))
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot Training vs Validation Loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Plot Training vs Validation Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# 7. SHAP Explainability
masker = shap.maskers.Independent(X_train[:100])
explainer = shap.Explainer(model, masker)
shap_values = explainer(X_test[:100])

shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

# 8. Save the Model
model.save('results/optimized_model.keras')

print("✅ Training Complete with Best Hyperparameters!")
