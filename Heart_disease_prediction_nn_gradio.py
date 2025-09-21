# Heart Disease Prediction using Neural Network and Gradio

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import gradio as gr

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv("heart.csv")

# Data preprocessing
# Handle missing values
dataset = dataset.dropna()

# Outlier clipping for numerical features
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in num_cols:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    dataset[col] = dataset[col].clip(lower_bound, upper_bound)

# Separate features and target
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(predictors)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, target, test_size=0.20, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Neural Network Model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with custom learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
model.fit(X_train, Y_train, epochs=300, batch_size=16, validation_split=0.2, 
          class_weight=class_weight_dict, callbacks=[early_stopping], verbose=0)

# Evaluate accuracy
Y_pred_prob = model.predict(X_test)
# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
Y_pred_nn = [1 if x[0] >= optimal_threshold else 0 for x in Y_pred_prob]
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print(f"The accuracy score achieved using Neural Network is: {score_nn} %")

# Prediction function for Gradio
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    try:
        # Convert inputs to appropriate types
        input_data = np.array([[float(age), int(sex), int(cp), float(trestbps), float(chol), int(fbs),
                                int(restecg), float(thalach), int(exang), float(oldpeak), int(slope),
                                int(ca), int(thal)]])
        
        # Clip input values to match training data preprocessing
        input_data[0, 0] = np.clip(input_data[0, 0], dataset['age'].min(), dataset['age'].max())  # Age
        input_data[0, 3] = np.clip(input_data[0, 3], dataset['trestbps'].min(), dataset['trestbps'].max())  # trestbps
        input_data[0, 4] = np.clip(input_data[0, 4], dataset['chol'].min(), dataset['chol'].max())  # chol
        input_data[0, 7] = np.clip(input_data[0, 7], dataset['thalach'].min(), dataset['thalach'].max())  # thalach
        input_data[0, 9] = np.clip(input_data[0, 9], dataset['oldpeak'].min(), dataset['oldpeak'].max())  # oldpeak
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        pred = model.predict(input_data_scaled)
        probability = pred[0][0]
        result = "Heart Disease" if probability >= optimal_threshold else "No Heart Disease"
        
        # Warning for high-risk inputs
        high_risk = (age > 60 or trestbps > 140 or chol > 240 or exang == 1 or ca >= 2 or thal == 7)
        warning = "Warning: High-risk factors detected (e.g., high age, cholesterol, or major vessels affected)." if high_risk else ""
        
        return f"Prediction: {result} (Probability: {probability:.2%})\n{warning}"
    except Exception as e:
        return f"Error: Invalid input. Please ensure all fields are filled correctly. ({str(e)})"

# Gradio Interface
inputs = [
    gr.Number(label="Age", value=50, minimum=0, maximum=120),
    gr.Radio([1, 0], label="Sex (1: Male, 0: Female)", value=1),
    gr.Radio([0, 1, 2, 3], label="Chest Pain Type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", value=0),
    gr.Number(label="Resting Blood Pressure (mm Hg)", value=120, minimum=0),
    gr.Number(label="Serum Cholesterol (mg/dl)", value=200, minimum=0),
    gr.Radio([1, 0], label="Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)", value=0),
    gr.Radio([0, 1, 2], label="Resting ECG Results (0, 1, 2)", value=0),
    gr.Number(label="Maximum Heart Rate Achieved", value=150, minimum=0),
    gr.Radio([1, 0], label="Exercise Induced Angina (1: Yes, 0: No)", value=0),
    gr.Number(label="Oldpeak (ST depression induced by exercise)", value=0.0, minimum=0),
    gr.Radio([0, 1, 2], label="Slope of Peak Exercise ST Segment", value=0),
    gr.Radio([0, 1, 2, 3, 4], label="Number of Major Vessels (0-4)", value=0),
    gr.Radio([3, 6, 7], label="Thal (3: normal, 6: fixed defect, 7: reversible defect)", value=3)
]

gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Prediction",
    description="Enter patient details to predict heart disease using an optimized Neural Network model. High-risk inputs (e.g., age > 60, high cholesterol, or major vessels affected) will trigger a warning."
).launch()