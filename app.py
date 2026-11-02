from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# Setup page title and layout
st.set_page_config(
    page_title="Hotel Reservation Classifier",
    layout="wide",
)
st.title("Hotel Reservation Classifier")


# Folder of model relative to current project
model_folder = Path("model")
meta_path = model_folder / "meta.joblib"
# Load model files from repository
if not meta_path.exists():
    raise FileNotFoundError(
        "Missing model/meta.joblib. Run python hotel_reservation_training.py first"
    )

# Load model details
meta_info = joblib.load(meta_path)
preprocessor = joblib.load(model_folder / meta_info["preprocessor_file"])
models = {}
for model_name, model_file in meta_info["model_files"].items():
    path = model_folder / model_file
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model file: {path}")
    models[model_name] = joblib.load(path)
    

# Setup side bar to select ML models
with st.sidebar:
    st.header("Available models")
    select_model = st.selectbox(
        "Model selection",
        options=list(models.keys()),
    )


# Upload option to user for test data
st.write("Upload a test CSV file and select a model")
uploaded_file = st.file_uploader("upload test data CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload test_data.csv to see metrics")
    st.stop()


# Read CSV file uploaded and metadata
test_data = pd.read_csv(uploaded_file)
features = meta_info["feature"]
target = meta_info["target"]
class_names = meta_info["class_names"]
encoding = meta_info["encoding"]


# Validate uploaded file
missing_column = [column for column in features + [target] if column not in test_data.columns]
if missing_column:
    st.error(f"Upload file is missing columns: {missing_column}")
    st.stop()
y_data = test_data[target].astype(str)
unknown_labels = sorted(set(y_data.unique()) - set(class_names))
if unknown_labels:
    st.error(f"Unknown target labels {unknown_labels}. Allowed values: {class_names}")
    st.stop()
X_test = test_data[features].copy()
y_test = y_data.map(encoding)


# Predict and Evalualuate model selected from UI screen
features_preprocessor = preprocessor.transform(X_test)
y_predictions = models[select_model].predict(features_preprocessor)
y_probabilities = models[select_model].predict_proba(features_preprocessor)[:, 1]

accuracy = float(accuracy_score(y_test, y_predictions))
auc = float(roc_auc_score(y_test, y_probabilities))
precision = float(precision_score(y_test, y_predictions, zero_division=0))
recall = float(recall_score(y_test, y_predictions, zero_division=0))
f1 = float(f1_score(y_test, y_predictions, zero_division=0))
mcc = float(matthews_corrcoef(y_test, y_predictions))
confusion = confusion_matrix(y_test, y_predictions)
    

# Display Evaluation metrics
st.subheader(f"Evaluation metrics - {select_model}")
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")


left, right = st.columns(2)

# Display left column of screen with Confusion Matrix
with left:
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
        annot_kws={"size": 9},
    )
    ax.set_title("Confusion Matrix", fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)    
    st.pyplot(fig, clear_figure=True, width="content")
    plt.close(fig)


# Display left column of screen with Classification report
with right:
    st.markdown("**classification report**")
    st.code(
        classification_report(
            y_test,
            y_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

