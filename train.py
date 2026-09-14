import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load data
df = pd.read_csv('Churn_Modelling.csv')

# Clean
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

# Split features/target
X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale — transform only on test, not fit_transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=11))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["accuracy"])

# Train
history = model.fit(X_train_scaled, y_train, epochs=120, validation_split=0.2)

# Evaluate
z = model.predict(X_test_scaled)
y_pred = np.where(z > 0.5, 1, 0)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model + scaler
model.save("model/churn_model.keras")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved.")