from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv(r'E:\4th_year\2nd Semester\PatternRecoginition\assignment_dataset\assignment\assignment1dataset.csv')
df = pd.DataFrame(data)

# Compute correlation and select features
correlation_matrix = df.corr()
target_correlation = correlation_matrix["RevenuePerDay"].abs().sort_values(ascending=False)
selected_features = target_correlation[target_correlation > 0.5].index.tolist()
selected_feature_dataset = df[selected_features]

def check_duplicates(df, column):
    """Check if a column is a duplicate using hashing (efficient check)."""
    column_hashes = {hash(tuple(df[c])) for c in df.columns}
    return hash(tuple(column)) not in column_hashes

def dynamic_degress(deg, df):
    """Generate polynomial features up to degree 'deg' using custom logic."""
    new_df = df.copy()
    for i in range(deg - 1):  # Degree -1 because original features are degree 1
        cols = new_df.columns
        for j in df.columns:
            for k in cols:
                tmp = df[j] * new_df[k]
                if check_duplicates(new_df, tmp):
                    new_col_name = f"{j} * {k}"
                    new_df[new_col_name] = tmp
    return new_df

# Lists to store errors
train_error_lst = [] 
test_error_lst = [] 
degree_lst = [] 

for i in range(1, 6):  # Try degrees from 1 to 5
    X_poly = dynamic_degress(i, selected_feature_dataset.drop(columns=['RevenuePerDay']).copy())
    y = selected_feature_dataset['RevenuePerDay']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Train polynomial regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Store values
    train_error_lst.append(train_mse)
    test_error_lst.append(test_mse)
    degree_lst.append(i)

# Plot Training vs Testing Error
plt.figure(figsize=(8, 6))
plt.plot(degree_lst, train_error_lst, marker='o', linestyle='-', color='blue', label="Training Error")
plt.plot(degree_lst, test_error_lst, marker='o', linestyle='-', color='red', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training vs Testing Error Across Polynomial Degrees")
plt.xticks(degree_lst)  # Ensures correct tick labels
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
