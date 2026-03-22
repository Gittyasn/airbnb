import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up Directories
if not os.path.exists('plots'):
    os.makedirs('plots')

# --- STEP 2: LOAD DATA ---
print("\n--- STEP 2: LOAD DATA ---")
data_path = 'data/AirBNB.csv'
df = pd.read_csv(data_path, low_memory=False)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show dataset shape
print(f"\nShape: {df.shape}")

# Column list
print("\nColumns:", df.columns.tolist())

# --- STEP 3: UNDERSTAND DATA ---
print("\n--- STEP 3: UNDERSTAND DATA ---")
df.info()
print("\nSummary statistics:")
print(df.describe())

# --- STEP 4: DATA CLEANING ---
print("\n--- STEP 4: DATA CLEANING ---")
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Dropping unnecessary columns
cols_to_drop = ['id', 'host_since']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Convert target and numerical features to numeric
df['log_price'] = pd.to_numeric(df['log_price'], errors='coerce')
df.dropna(subset=['log_price'], inplace=True)

num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_rating']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Check duplicates and remove if any
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)

# --- CATEGORICAL FIX ---
# Critical Fix: Convert all categorical columns to string to avoid mixed type errors (bool vs str)
cat_cols = ['room_type', 'cancellation_policy', 'city', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cleaning_fee']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).replace('nan', 'missing')

# --- STEP 5: BASIC EDA ---
print("\n--- STEP 5: BASIC EDA ---")

# Histogram for log_price
plt.figure(figsize=(10, 6))
sns.histplot(df['log_price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Log Price')
plt.xlabel('Log Price')
plt.ylabel('Frequency')
plt.savefig('plots/log_price_distribution.png')
print("Saved plots/log_price_distribution.png")

# Count plot for room_type
if 'room_type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=df, palette='viridis')
    plt.title('Count of Room Types')
    plt.savefig('plots/room_type_count.png')
    print("Saved plots/room_type_count.png")

# --- STEP 6: OUTLIER DETECTION ---
print("\n--- STEP 6: OUTLIER DETECTION ---")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['log_price'], color='salmon')
plt.title('Log Price Outliers')
plt.savefig('plots/log_price_boxplot.png')
print("Saved plots/log_price_boxplot.png")

# --- STEP 7: BIVARIATE ANALYSIS ---
print("\n--- STEP 7: BIVARIATE ANALYSIS ---")

# Price vs room_type
if 'room_type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='room_type', y='log_price', data=df)
    plt.title('Log Price vs Room Type')
    plt.savefig('plots/price_vs_roomtype.png')
    print("Saved plots/price_vs_roomtype.png")

# --- STEP 8: FEATURE ENGINEERING ---
print("\n--- STEP 8: FEATURE ENGINEERING ---")

# --- STEP 9: FEATURE SELECTION ---
print("\n--- STEP 9: FEATURE SELECTION ---")
numerical_features = [col for col in num_cols if col in df.columns]
categorical_features = [col for col in cat_cols if col in df.columns]
X = df[numerical_features + categorical_features]
y = df['log_price']

# Correlation heatmap (numeric only)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')
print("Saved plots/correlation_heatmap.png")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 10: SIMPLE LINEAR REGRESSION ---
print("\n--- STEP 10: SIMPLE LINEAR REGRESSION ---")
# Using a Pipeline to handle NaNs automatically
simple_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', LinearRegression())
])
X_sm_train = X_train[['accommodates']]
X_sm_test = X_test[['accommodates']]
simple_pipeline.fit(X_sm_train, y_train)
y_pred_sm = simple_pipeline.predict(X_sm_test)

print(f"Simple LR R2 Score: {r2_score(y_test, y_pred_sm):.4f}")

# --- STEP 11: MULTIPLE LINEAR REGRESSION ---
print("\n--- STEP 11: MULTIPLE LINEAR REGRESSION ---")

# Pipeline components
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

multiple_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

multiple_model.fit(X_train, y_train)
y_pred_m = multiple_model.predict(X_test)

r2_m = r2_score(y_test, y_pred_m)
print(f"Multiple LR R2: {r2_m:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_m):.4f}")

# --- STEP 12: MODEL COMPARISON ---
print("\n--- STEP 12: MODEL COMPARISON ---")
print(f"Simple Linear Regression R2: {r2_score(y_test, y_pred_sm):.4f}")
print(f"Multiple Linear Regression R2: {r2_m:.4f}")

# --- STEP 13: PREDICTION ---
print("\n--- STEP 13: PREDICTION ---")
sample_data = X_test.iloc[[0]]
predicted_price = multiple_model.predict(sample_data)[0]
print(f"Actual price(log): {y_test.iloc[0]:.4f}")
print(f"Predicted price(log): {predicted_price:.4f}")

# --- STEP 14: VISUALIZATION ---
print("\n--- STEP 14: VISUALIZATION ---")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_m, alpha=0.3, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted Weights')
plt.savefig('plots/actual_vs_predicted.png')
plt.close()

# Save the model
joblib.dump(multiple_model, 'src/model.pkl')
print("Model saved to src/model.pkl")

# --- STEP 15: FINAL OUTPUT ---
print("\n--- STEP 15: FINAL OUTPUT ---")
print(f"Successful project completion. Multiple LR R2: {r2_m:.4f}")
print(f"Final Prediction Example: ${np.exp(predicted_price):.2f}")
