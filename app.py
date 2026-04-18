import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Uber Trip Analytics | Predictor",
    page_icon="🚕",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM THEMING & UI ENHANCEMENTS ---
st.markdown("""
    <style>
    /* Main Container Styling */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 42px;
        color: #1DB954; /* Emerald Green */
    }
    /* Button Customization */
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #000000;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1DB954;
        color: white;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING (OPTIMIZED) ---
@st.cache_resource(show_spinner="Loading Predictive Engine...")
def load_predictive_assets():
    try:
        with open('uber_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('model_columns.pkl', 'rb') as columns_file:
            columns = pickle.load(columns_file)
        return model, columns
    except FileNotFoundError:
        st.error("Critical Error: Predictive assets (.pkl files) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# Initialize Engine
model, model_columns = load_predictive_assets()

# --- HEADER SECTION ---
st.title("🚕 Uber Trip Distance Predictor")
st.markdown("""
    **Objective:** Utilize advanced Gradient Boosting (XGBoost) to estimate trip mileage based on temporal and categorical features. 
    Validated Model Accuracy: **82.1% (R² Score)**.
""")
st.divider()

# --- INPUT INTERFACE ---
st.subheader("📋 Ride Parameters")

# Sidebar Configuration for Secondary Metadata
st.sidebar.title("Configuration")
st.sidebar.info("This system utilizes One-Hot Encoding and IQR-capped distance metrics for high-fidelity predictions.")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Trip Category", ["Business", "Personal"])
        purpose = st.selectbox("Purpose of Trip", [
            "Meeting", "Meal/Entertain", "Errand/Supplies", "Customer Visit", 
            "Temporary Site", "Between Offices", "Charity ($)", "Commute", "Moving", "Airport/Travel", "Missing"
        ])
        duration = st.number_input("Duration (Minutes)", min_value=1.0, step=0.5, value=15.0, help="Total estimated time spent in transit.")

    with col2:
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        month = st.selectbox("Month of Travel", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        time_label = st.selectbox("Time Window", ["Morning", "Afternoon", "Evening", "Night"])

# --- PREDICTION PIPELINE ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Generate Trip Forecast"):
    with st.spinner("Executing Prediction Pipeline..."):
        # Construct Input Vector
        input_vector = pd.DataFrame(columns=model_columns)
        input_vector.loc[0] = 0  # Initialize with zero vector
        
        # Primary Feature Assignment
        input_vector.at[0, 'duration'] = duration
        
        # One-Hot Feature Mapping
        categorical_mappings = {
            'category': category,
            'purpose': purpose,
            'day_name': day,
            'month': month,
            'time_label': time_label
        }
        
        for feature, selection in categorical_mappings.items():
            encoded_col = f"{feature}_{selection}"
            if encoded_col in model_columns:
                input_vector.at[0, encoded_col] = 1
        
        # Execution
        try:
            prediction = model.predict(input_vector)[0]
            
            # --- RESULTS DISPLAY ---
            st.divider()
            r_col1, r_col2 = st.columns([2, 1])
            
            with r_col1:
                st.metric(label="Predicted Displacement", value=f"{prediction:.2f} Miles")
                
            with r_col2:
                if prediction > 15:
                    st.warning("Distance Alert: High-Mileage Trip Detected.")
                else:
                    st.success("Distance Alert: Short-Haul Trip Detected.")
            
            # Data Integrity Check (Internal Log)
            st.caption(f"Input Features Processed: {len(model_columns)} dimensions.")
            
        except Exception as e:
            st.error(f"Prediction Pipeline Failure: {e}")

# --- FOOTER SECTION ---
st.divider()
st.markdown("""
    <div style="text-align: center; color: #888888; font-size: 14px;">
        Yash | Bachelor of Technology, CSE | DCRUST Murthal<br>
        Machine Learning Portfolio Project © 2026
    </div>
""", unsafe_allow_html=True)