# Install Required Libraries
# pip install xgboost

# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xgboost

# Data Preprocessing and Model Selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
from sklearn import metrics
import pickle

# Step 1: Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
df = train.copy()
df.drop(["Outlet_Size", "Item_Weight", "Item_Identifier", "Outlet_Identifier"], axis=1, inplace=True)

# Encoding Categorical Variables
# Encoding Categorical Variables
fat = {"Low Fat": 0, "Regular": 1, "low fat": 0, "LF": 0, "reg": 1}
df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat)

tier = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map(tier)

market_type = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
df['Outlet_Type'] = df['Outlet_Type'].map(market_type)

# Ensure to encode 'Item_Type' column
item_type_mapping = {item: i for i, item in enumerate(df['Item_Type'].unique())}
df['Item_Type'] = df['Item_Type'].map(item_type_mapping)

df['Age_Outlet'] = 2021 - df['Outlet_Establishment_Year']
df.drop("Outlet_Establishment_Year", axis=1, inplace=True)

# Handling Outliers in Item Visibility
q3, q1 = np.percentile(df["Item_Visibility"], [75, 25])
iqr = q3 - q1
df.loc[df["Item_Visibility"] > 1.5 * iqr, "Item_Visibility"] = 0.066132


# Step 3: Split Dataset
x = df.drop("Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=400, min_samples_split=8, min_samples_leaf=100, max_depth=6),
    "Decision Tree Regression": DecisionTreeRegressor(max_depth=15, min_samples_leaf=100),
    "Gradient Boosting Regression": GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, min_samples_split=8, min_samples_leaf=100),
    "AdaBoost Regression": AdaBoostRegressor(n_estimators=1000, learning_rate=0.01),
    "XGB Regression": XGBRFRegressor(n_jobs=-1, n_estimators=1000, max_depth=5),
    "MLP Regression": MLPRegressor(),
    "SVR": SVR(kernel='linear', C=10, gamma='scale')
}

# Step 5: Model Training and Evaluation
accuracy = {}
rmse = {}
explained_variance = {}
max_error = {}
MAE = {}

def train_model(model, model_name):
    print(f"Training Model: {model_name}")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    accuracy[model_name] = metrics.r2_score(y_test, pred) * 100
    rmse[model_name] = np.sqrt(metrics.mean_squared_error(y_test, pred))
    explained_variance[model_name] = metrics.explained_variance_score(y_test, pred)
    max_error[model_name] = metrics.max_error(y_test, pred)
    MAE[model_name] = metrics.mean_absolute_error(y_test, pred)

    print(f"R2_Score: {accuracy[model_name]:.2f}%")
    print(f"RMSE: {rmse[model_name]:.2f}")
    print(f"Explained Variance: {explained_variance[model_name]:.2f}")
    print(f"Max Error: {max_error[model_name]:.2f}")
    print(f"Mean Absolute Error: {MAE[model_name]:.2f}\n")

# Train and Evaluate Each Model
for name, model in models.items():
    train_model(model, name)

# Step 6: Save the Best Model (Assuming Random Forest performed the best)
best_model = models["Random Forest Regression"]
with open('model2.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Best model saved as 'model2.pkl'")
