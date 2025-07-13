
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load dataset
df = pd.read_csv('HistoricalQuotes.csv')

# Step 2: Convert 'Date' column if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

# Step 3: Clean 'Close' column
close_column = [col for col in df.columns if 'close' in col.lower()][0]
data = df[[close_column]].dropna()

# Remove $, commas, and convert to float
data[close_column] = data[close_column].replace('[\$,]', '', regex=True).astype(float)

# Step 4: Normalize prices
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Create sequences
window = 60
X, y = [], []
for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Step 6: Train-test split for ML models
X_flat = X.reshape((X.shape[0], X.shape[1]))
X_train_f, X_test_f, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 7: Split for LSTM
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))

# Step 8: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_f, y_train)
lr_preds = lr_model.predict(X_test_f)

# Step 9: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_f, y_train.ravel())
rf_preds = rf_model.predict(X_test_f)

# Step 10: LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=1)
lstm_preds = model.predict(X_test_seq)

# Step 11: Plotting function with labels
def plot_results(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Prices (Test Data)', linewidth=2)
    plt.plot(predicted, '--', label='Predicted Prices', linewidth=2)
    plt.title(f"{title} - Actual vs Predicted Prices (Test Set)", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Normalized Price", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 12: Show plots
plot_results(y_test, lr_preds, 'Linear Regression')
plot_results(y_test, rf_preds, 'Random Forest')
plot_results(y_test_seq, lstm_preds, 'LSTM')

# Step 13: Show evaluation metrics
print("📉 Linear Regression R² Score:", r2_score(y_test, lr_preds))
print("🌲 Random Forest R² Score:", r2_score(y_test, rf_preds))
print("🔁 LSTM Mean Squared Error:", mean_squared_error(y_test_seq, lstm_preds))