from pathlib import Path
import joblib
import streamlit as st


# Directory of model details
model_folder = Path("model")
meta_path = model_folder / "meta.joblib"

st.set_page_config(
    page_title="Hotel Reservation",
    layout="wide",
)

st.title("Hotel Reservation Classifier")


# Load model files from repository
@st.cache_resource
def load_runtime_bundle():
    
    if not meta_path.exists():
        raise FileNotFoundError(
            "Missing model/meta.joblib. Run python hotel_reservation_training.py first"
        )
    
    training_meta_info = joblib.load(meta_path)
    preprocessor = joblib.load(model_folder / training_meta_info["preprocessor_file"])

    models = {}
    for model_name, model_file in training_meta_info["model_files"].items():
        path = model_folder / model_file
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model file: {path}")
        models[model_name] = joblib.load(path)
    
    return training_meta_info, preprocessor, models


# Load model files and display to user
try:
    meta_info, preprocessor, models = load_runtime_bundle()

except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

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

    