import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib 
import tkinter as tk 
from tkinter import messagebox
import unittest

# -------------------------------
# Load and prepare the data
# -------------------------------

file_path = r'C:\Users\nicho\Downloads\Net_Worth_Data.xlsx'
data = pd.read_excel(file_path)

input_df = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth'], axis=1)
output_df = data['Net Worth']

# ---------------------------------
# S.R 1.4
# Print the first few rows
# Print the last few rows
# Print the shape of data
# Print a concise summary of the data set
# ---------------------------------

#shows top 5 rows
print("top 5 rows:\n", data.head())

#shows bottom 5 rows
print("bottom 5 rows:\n", data.tail())

#shows shape of data
print("num row&col\n", data.shape)
print("num row\n", data.shape[0])
print("num col\n", data.shape[1])

#concise summary:
print(data.info())

# ---------------------------------
# S.R 1.6
# Train and test the identified 10 models to evaluate their 
# performance and determine which model provides the most accurate predictions.
# ---------------------------------

# ---------------------------------
# Preprocessing: Normalize and Split
# ---------------------------------

# Handle categorical columns (if any) using one-hot encoding
input_df_encoded = pd.get_dummies(input_df, drop_first=True)

# Fill missing values if any
input_df_encoded = input_df_encoded.fillna(input_df_encoded.mean())

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(input_df_encoded)

# Save feature names for later use
feature_names = input_df_encoded.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, output_df, test_size=0.2, random_state=42
)

joblib.dump(feature_names, 'feature_names.pkl')

# ---------------------------------
# Initialize 10 Models
# ---------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(verbosity=0),
    "LightGBM": LGBMRegressor(),
    "SVR": SVR(),
    "KNN Regressor": KNeighborsRegressor()
}

# ---------------------------------
# Train, Predict, Evaluate
# ---------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append((name, r2, mae))
    print(f"\n{name}:\nR² Score: {r2:.4f}\nMAE: {mae:.2f}")

# ---------------------------------
# Display Best Model
# ---------------------------------

best_model = max(results, key=lambda x: x[1])  # Based on highest R²
print(f"\nBest Model: {best_model[0]} with R² Score = {best_model[1]:.4f} and MAE = {best_model[2]:.2f}")

# ---------------------------------
# S.R 1.7
# Visualize the results of the models with a bar chart
# ---------------------------------

# Extract model names and their R² scores
model_names = [result[0] for result in results]
r2_scores = [result[1] for result in results]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=r2_scores, y=model_names, palette="viridis")
plt.xlabel('R² Score')
plt.title('Model Performance Comparison - R² Score')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# ---------------------------------
# S.R 1.8
# Save best model and scaler
# ---------------------------------

best_model_name = best_model[0]
saved_model = models[best_model_name]
joblib.dump(saved_model, 'best_net_worth_model.joblib')
joblib.dump(scaler, 'input_scaler.joblib')

# ---------------------------------
# S.R 1.8
# Load best model and scaler, predict Net Worth for a custom input
# ---------------------------------

loaded_model = joblib.load('best_net_worth_model.joblib')
loaded_scaler = joblib.load('input_scaler.joblib')
feature_names = joblib.load('feature_names.pkl') 

# Create a sample input dictionary for prediction
# Must match the encoded feature columns exactly
sample_input = {col: 0 for col in input_df_encoded.columns}  # default zeros

# Update sample_input with example realistic values
sample_input['Age'] = 40
sample_input['Income'] = 85000
sample_input['Credit Card Debt'] = 5000
sample_input['Healthcare Cost'] = 2000
sample_input['Inherited Amount'] = 10000
sample_input['Stocks'] = 15000
sample_input['Bonds'] = 8000
sample_input['Mutual Funds'] = 12000
sample_input['ETFs'] = 7000
sample_input['REITs'] = 3000

# Convert to DataFrame and scale
sample_df = pd.DataFrame([[sample_input.get(col, 0) for col in feature_names]], columns=feature_names)
sample_scaled = loaded_scaler.transform(sample_df)

# Predict net worth
predicted_net_worth = loaded_model.predict(sample_scaled)
print(f"\nPredicted Net Worth for sample input: ${predicted_net_worth[0]:,.2f}")

# ---------------------------------
# S.R 1.9
# Create GUI with Tkinter for user input
# ---------------------------------

# GUI Prediction Function
def predict_net_worth():
    try:
        user_input = {}
        for feature, entry in entries.items():
            val = entry.get()
            user_input[feature] = float(val) if val.strip() else 0.0  # Use 0 if field is empty

        # Recreate the input row in the correct column order
        user_input_df = pd.DataFrame([[user_input.get(col, 0) for col in feature_names]], columns=feature_names)
        user_input_scaled = loaded_scaler.transform(user_input_df)

        # Predict and show result
        predicted_worth = loaded_model.predict(user_input_scaled)
        messagebox.showinfo("Estimated Net Worth",
                            f"💰 Predicted Net Worth: ${predicted_worth[0]:,.2f}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")

# GUI Setup
window = tk.Tk()
window.title("Net Worth Estimator")

entries = {}

# Create input fields dynamically
for idx, feature in enumerate(feature_names):
    label = tk.Label(window, text=feature.replace('_', ' ').capitalize() + ":")
    label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')

    entry = tk.Entry(window)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = entry

# Predict Button
predict_btn = tk.Button(window, text="Estimate Net Worth", command=predict_net_worth)
predict_btn.grid(row=len(feature_names), columnspan=2, pady=15)

# Start the GUI loop
window.mainloop()

# ---------------------------------
# S.R 2.1 - 2.3
# Unit tests for privacy principles
# ---------------------------------

# -----------------------------------
# S.R 2.1   
# Unit tests for input privacy principles
# -----------------------------------
class TestPrivacyPrinciples(unittest.TestCase):
    
    def setUp(self):
        file_path = r'C:\Users\nicho\Downloads\Net_Worth_Data.xlsx'
        self.data = pd.read_excel(file_path)

    def test_input_data_is_anonymized(self):
        pii_columns = ['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth']
        for col in pii_columns:
            self.assertIn(col, self.data.columns, f"Expected PII column '{col}' missing from raw data.")
        processed_input_df = self.data.drop(pii_columns, axis=1)
        for col in pii_columns:
            self.assertNotIn(col, processed_input_df.columns,
                             f"Privacy violation: '{col}' should not be in the input dataframe!")

# ---------------------------------
# S.R 2.2
# Unit tests for output privacy principles
# ---------------------------------
class TestOutputPrivacyPrinciples(unittest.TestCase):

    def setUp(self):
        self.file_path = r'C:\Users\nicho\Downloads\Net_Worth_Data.xlsx'
        self.data = pd.read_excel(self.file_path)
        input_df = self.data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth'], axis=1)
        input_df_encoded = pd.get_dummies(input_df, drop_first=True)
        input_df_encoded.fillna(input_df_encoded.mean(), inplace=True)
        self.scaler = joblib.load('input_scaler.joblib')
        self.model = joblib.load('best_net_worth_model.joblib')
        self.feature_names = joblib.load('feature_names.pkl')
        self.sample = pd.DataFrame([[0 for _ in self.feature_names]], columns=self.feature_names)
        self.sample_scaled = self.scaler.transform(self.sample)

    def test_output_is_anonymized(self):
        prediction = self.model.predict(self.sample_scaled)
        self.assertTrue(isinstance(prediction, (np.ndarray, list, float)), "Output should be numeric, not structured with personal info.")
        output_str = str(prediction)
        pii_indicators = ['@', 'Client Name', 'e-mail']
        for pii in pii_indicators:
            self.assertNotIn(pii, output_str, f"Privacy breach: Output includes personal data like '{pii}'.")

# ---------------------------------
# S.R 2.3
# Unit tests for data shape privacy principles
# ---------------------------------
class TestDatasetShapePrivacy(unittest.TestCase):
    
    def setUp(self):
        self.file_path = r'C:\Users\nicho\Downloads\Net_Worth_Data.xlsx'
        self.data = pd.read_excel(self.file_path)
        self.data = self.data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country'], axis=1)
        self.expected_columns = {
            'Age', 'Income', 'Credit Card Debt', 'Healthcare Cost', 'Inherited Amount',
            'Stocks', 'Bonds', 'Mutual Funds', 'ETFs', 'REITs'
        }
        self.sensitive_columns = {
            'Client Name', 'Client e-mail', 'Profession', 'Education', 'Country'
        }

    def test_expected_columns_present(self):
        current_columns = set(self.data.columns)
        for col in self.expected_columns:
            self.assertIn(col, current_columns, f"Missing expected column: {col}")

    def test_no_sensitive_columns_present(self):
        current_columns = set(self.data.columns)
        for sensitive_col in self.sensitive_columns:
            self.assertNotIn(sensitive_col, current_columns, f"Privacy breach: Found sensitive column '{sensitive_col}'")

    def test_column_count(self):
        self.assertEqual(
            len(self.expected_columns),
            len(self.data[list(self.expected_columns)].columns),
            "Unexpected number of columns. Dataset may contain extra or missing columns."
        )

if __name__ == "__main__":
    unittest.main()
