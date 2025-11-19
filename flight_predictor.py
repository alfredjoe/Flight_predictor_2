# -*- coding: utf-8 -*-
"""Flight Predictor - Production Version"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load data
data = pd.read_csv("flight_delay_clean.csv")

# Create target variable: 1 if arrival delay > 15 minutes, else 0
data['ArrDel15'] = (data['ArrDelay'] > 15).astype(int)

# Create departure delay indicator
data['DepDel15'] = (data['DepDelay'] > 15).astype(int)

# Feature engineering
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x in [1, 7] else 0)

# Extract date features from FlightDate
data['FlightDate'] = data['FlightDate'].astype(str)
data['Month'] = data['FlightDate'].str[4:6].astype(int)
data['Day'] = data['FlightDate'].str[6:8].astype(int)

# Select features for modeling
db = data[['DayOfWeek', 'Origin', 'Dest', 'DepDelay', 'IsWeekend', 'Month', 'Day', 'DepDel15', 'ArrDel15']].copy()

# Encode categorical variables and save encoders
label_encoders = {}
for col in ['Origin', 'Dest']:
    le = LabelEncoder()
    db[col] = le.fit_transform(db[col])
    label_encoders[col] = le

# Separate features and target
y = db['ArrDel15']
X = db.drop(columns=['ArrDel15'])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Save models and preprocessing objects
joblib.dump(lr_model, 'flight_predictor_lr.joblib')
joblib.dump(knn_model, 'flight_predictor_knn.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print("Models saved successfully!")