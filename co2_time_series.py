




import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset from a CSV file
file_path = '/Users/jeb_bryson12/Documents/School Documents/Spring 2024/STOR 556/Time Series Python/co2 copy.csv'  # Update this to the path of your CSV file
df = pd.read_csv(file_path)

# Combine 'year' and 'month' into a single 'date' column
df['date'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])

# Feature Engineering: Creating lag features
df.sort_values('date', inplace=True)
for i in range(1, 13):
    df[f'lag_{i}'] = df['average'].shift(i)

# Remove rows with NaN values (the first 12 months now have missing lagged values)
df.dropna(inplace=True)

# Define features and target
X = df[[f'lag_{i}' for i in range(1, 13)]]
y = df['average']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# Example: Predict the next month's CO2 level (you need to replace this part with your actual data)
# Assuming 'df' has the latest 12 months of data at the bottom
next_month_features = df[[f'lag_{i}' for i in range(1, 13)]].iloc[-1].shift(-1)
next_month_features['lag_1'] = df['average'].iloc[-1]  # Last known value as the most recent lag
next_month_prediction = model.predict(next_month_features.values.reshape(1, -1))

print(f"Predicted CO2 level for next month: {next_month_prediction[0]}")