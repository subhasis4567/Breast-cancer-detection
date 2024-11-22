import streamlit as st
import pickle
import numpy as np

# Load the trained models
with open("combined_models.pkl", 'rb') as file:
    models = pickle.load(file)

# Set the background image and overlay styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://as2.ftcdn.net/v2/jpg/06/94/25/69/1000_F_694256960_ndnsbFkMzsC0UqhVZ1Zx0TXv0RhGs7ik.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }
    .overlay {
        background-color: rgba(0, 0, 0, 0.7);  /* Darker black overlay for better text visibility */
        padding: 20px;
        border-radius: 10px;
        color: white;  /* Text color */
        font-size: 18px;  /* Increased font size for better readability */
    }
    .title {
        font-size: 24px;  /* Title font size */
        font-weight: bold;
    }
    .header {
        font-size: 20px;  /* Header font size */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown('<div class="title">Breast Cancer Prediction App</div>', unsafe_allow_html=True)

# Add a container with a semi-transparent background
with st.container():
    st.markdown('<div class="overlay">', unsafe_allow_html=True)
    
    # Input fields for the user using sliders
    st.markdown('<div class="header">Input Features</div>', unsafe_allow_html=True)

    # Create a 3-column layout for sliders
    col1, col2, col3 = st.columns(3)

    with col1:
        clump_thickness = st.slider("Clump Thickness", min_value=1, max_value=10, value=5)
        uniformity_cell_size = st.slider("Uniformity of Cell Size", min_value=1, max_value=10, value=5)
        marginal_adhesion = st.slider("Marginal Adhesion", min_value=1, max_value=10, value=5)

    with col2:
        uniformity_cell_shape = st.slider("Uniformity of Cell Shape", min_value=1, max_value=10, value=5)
        single_epithelial_cell_size = st.slider("Single Epithelial Cell Size", min_value=1, max_value=10, value=5)
        bland_chromatin = st.slider("Bland Chromatin", min_value=1, max_value=10, value=5)

    with col3:
        bare_nuclei = st.slider("Bare Nuclei", min_value=1, max_value=10, value=5)
        normal_nucleoli = st.slider("Normal Nucleoli", min_value=1, max_value=10, value=5)
        mitoses = st.slider("Mitoses", min_value=1, max_value=10, value=5)

    # Create a DataFrame with the input values
    input_data = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape, 
                            marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
                            bland_chromatin, normal_nucleoli, mitoses]])

    # Model selection
    model_type = st.selectbox("Select Model for Prediction", ["SVM", "KNN", "Logistic Regression"])

    # Prediction button
    if st.button("Predict"):
        # Make predictions based on the selected model
        if model_type == "SVM":
            prediction = models['svm_model'].predict(input_data)
        elif model_type == "KNN":
            prediction = models['knn_model'].predict(input_data)
        else:
            prediction = models['logreg_model'].predict(input_data)

        # Display the prediction result
        if prediction[0] == 4:
            st.success("Predicted Class: Malignant (4)")
        else:
            st.success("Predicted Class: Benign (2)")

    st.markdown('</div>', unsafe_allow_html=True)

# Run the app footer
st.write("Developed by Subhasis")