# Preprocessing with LASSO


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from imblearn.combine import SMOTETomek

print("Starting Preprocessing...")

# 1. Load Dataset
df = pd.read_csv('../dataset/Final.csv')

# 2. Clean Data
df = df.dropna(axis=1, how='all').drop_duplicates()
df = df.select_dtypes(include=[np.number])

X = df.drop(columns=['Preg_Complication'], errors='ignore')
y = df['Preg_Complication']

# 3. Fill Missing Values
X = X.fillna(X.median())

# 4. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Feature Selection using LASSO
print("🧵 Running LASSO for Feature Selection...")
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
selected_features = np.where(lasso.coef_ != 0)[0]
X_selected = X_scaled[:, selected_features]

# Save Selected Feature Names
feature_names = X.columns[selected_features].to_list()
np.save('../results/feature_names.npy', feature_names)  # Save feature names

# 6. Save Selected Features for Reuse
np.save('../results/selected_features.npy', selected_features)

# 7. Handle Class Imbalance using NearSMOTE
print("Balancing Classes with NearSMOTE...")
smt = SMOTETomek(random_state=42)
X_balanced, y_balanced = smt.fit_resample(X_selected, y)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# 9. Save Processed Data for Model Training
np.save('../results/X_train.npy', X_train)
np.save('../results/X_test.npy', X_test)
np.save('../results/y_train.npy', y_train)
np.save('../results/y_test.npy', y_test)

print(f"Preprocessing Complete. Samples: {len(X_train)} train, {len(X_test)} test")
print("Data saved to 'results/' folder.")
