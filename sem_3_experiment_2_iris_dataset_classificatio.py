import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Load the Dataset
# Assuming the Kaggle dataset is named 'Iris.csv'
df = pd.read_csv('Iris.csv')

# The Kaggle dataset usually has an 'Id' column which we don't need for training
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 2. Preprocess the Data
X = df.drop('Species', axis=1).values
y = df['Species'].values

# Encode the target labels (e.g., 'Iris-setosa' -> 0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert labels to one-hot encoding for Categorical Cross-Entropy (CCE)
y_categorical = to_categorical(y_encoded)

# 3. Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Standardize the features (Neural networks perform better with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the MLP Model
model = Sequential()

# Input layer + First hidden layer (20 neurons). Using ReLU for hidden layers.
model.add(Dense(20, input_dim=4, activation='relu', name='Hidden_1'))

# Second hidden layer (20 neurons)
model.add(Dense(20, activation='relu', name='Hidden_2'))

# Output layer (3 neurons for 3 Iris species) using Softmax activation
model.add(Dense(3, activation='softmax', name='Output'))

# 5. Compile the Model
# We use Stochastic Gradient Descent (SGD) which becomes Mini-Batch GD 
# when we specify a batch_size during training.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', # CCE Loss
              metrics=['accuracy'])

# 6. Train the Model (Mini-batch Gradient Descent)
# A batch_size of 16 means the model updates weights after looking at 16 samples.
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.1, # Keep 10% of training data for validation tracking
    verbose=1
)

# 7. Make Predictions
y_pred_prob = model.predict(X_test)

# Convert probabilities back to class indices
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ---------------------------------------------------------
# 8. Plot Training Loss
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Categorical Cross-Entropy Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ---------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_,
            annot_kws={"size": 14})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.show()

# ---------------------------------------------------------
# 10. Classification Report
# ---------------------------------------------------------
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))
