
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
cancer = load_breast_cancer()

# Select four features
feature_indices = [0, 1, 2, 3]
X = cancer.data[:, feature_indices]
y = cancer.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# Streamlit Configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🎗️",
    layout="centered"
)

st.title("🎗️ Breast Cancer Classification")
st.write(
    "Enter the tumour measurements to predict "
    "whether the tumour is benign or malignant."
)

st.write(f"### Model Accuracy: {accuracy:.2%}")

st.divider()

# User Inputs
radius = st.number_input(
    "Mean Radius",
    min_value=0.0,
    value=14.0
)

texture = st.number_input(
    "Mean Texture",
    min_value=0.0,
    value=20.0
)

perimeter = st.number_input(
    "Mean Perimeter",
    min_value=0.0,
    value=90.0
)

area = st.number_input(
    "Mean Area",
    min_value=0.0,
    value=600.0
)

# Prediction
if st.button("Predict Tumour Class"):

    input_data = [[radius, texture, perimeter, area]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    result = cancer.target_names[prediction[0]]

    if result == "benign":
        st.success("Predicted Class: BENIGN")
    else:
        st.error("Predicted Class: MALIGNANT")

    st.subheader("Prediction Confidence")

    for i, name in enumerate(cancer.target_names):
        st.write(
            f"{name.upper()}: {probability[0][i]*100:.2f}%"
        )

    st.progress(float(max(probability[0])))

st.divider()

st.caption(
    "Developed using Streamlit and Scikit-Learn"
)
