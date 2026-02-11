import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the data
try:
    df = pd.read_csv('opensim_data.csv')
except FileNotFoundError:
    print("Error: 'opensim_data.csv' not found. Please run generate_data.py first.")
    exit()

X = df[['mass', 'initial_speed']]
y = df['final_position']

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define 5 Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR (RBF Kernel)": SVR(kernel='rbf'),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# 3. Evaluate and create Table
comparison_data = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    comparison_data.append({
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    })

# Convert to DataFrame for a clean table
results_df = comparison_data = pd.DataFrame(comparison_data)

# Report the best one based on R2 Score
best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]

print("\n--- Step 6: ML Model Comparison Table ---")
print(results_df.to_string(index=False))
print(f"\nConclusion: The best model is {best_model['Model']} with an R2 Score of {best_model['R2 Score']}.")

# 4. Generate Graph
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['Model'], results_df['R2 Score'], color=['skyblue', 'lightgreen', 'salmon', 'wheat', 'orchid'])

# Add labels and formatting
plt.xlabel('Machine Learning Models', fontsize=12)
plt.ylabel('R-Squared (Accuracy Score)', fontsize=12)
plt.title('Comparison of ML Models on OpenSim Data', fontsize=14)
plt.ylim(0, 1.1)  # R2 is max 1.0

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, yval, ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison_graph.png')
print("\nGraph saved as 'model_comparison_graph.png'. You can open this file to see your results!")