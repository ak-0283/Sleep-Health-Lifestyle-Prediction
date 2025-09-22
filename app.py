import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🛌 Sleep Disorder Prediction System")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV (Sleep Health Dataset)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # Encode categorical features
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])
    data['Occupation'] = encoder.fit_transform(data['Occupation'])
    data['BMI Category'] = encoder.fit_transform(data['BMI Category'])
    data['Sleep Disorder'] = encoder.fit_transform(data['Sleep Disorder'])

    # Fix Blood Pressure column
    bp_split = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic'] = bp_split[0].astype(float)
    data['Diastolic'] = bp_split[1].astype(float)
    data = data.drop('Blood Pressure', axis=1)

    # Features and Target
    X = data.drop(["Sleep Disorder"], axis=1)
    y = data["Sleep Disorder"]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Model
    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    # Model Evaluation
    if st.checkbox("Show Model Evaluation"):
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        st.write(f"📊 **Testing Accuracy:** {test_acc:.2f}%")
        
        st.text("Classification Report (Test Data):")
        st.text(classification_report(y_test, y_test_pred))

        # Confusion Matrix
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_test_pred),
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=encoder.inverse_transform([0,1,2]),
                    yticklabels=encoder.inverse_transform([0,1,2]), ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Sidebar for Custom Sample Input
    st.sidebar.header("Test a Custom Sample")
    default_sample = [101, 1, 30, 2, 6, 8, 2200, 8, 60, 72, 5000, 126, 83]
    user_input = []
    for i, col in enumerate(X.columns):  # <-- X is defined here, inside the block
        value = st.sidebar.number_input(f"{col}", value=default_sample[i])
        user_input.append(value)

    # Prediction Button
    if st.sidebar.button("Predict Sleep Disorder"):
        sample_df = pd.DataFrame([user_input], columns=X.columns)
        pred = model.predict(sample_df.values)
        st.markdown("### 🛌 Predicted Sleep Disorder")
        st.success(f"**{encoder.inverse_transform(pred)[0]}**")

else:
    st.warning("⚠️ Please upload the Sleep Health Dataset CSV to start.")