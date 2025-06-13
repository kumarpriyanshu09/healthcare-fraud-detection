import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Healthcare Fraud Detection - Test",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Fraud Detection - Test")
st.write("This is a test version to verify the app works.")

# Test basic functionality
st.success("✅ Basic Streamlit functionality working!")

# Test data loading
try:
    import plotly.express as px
    st.success("✅ Plotly imported successfully!")
except ImportError as e:
    st.error(f"❌ Plotly import failed: {e}")

# Test simple chart
fig_data = pd.DataFrame({
    'Category': ['Legitimate', 'Fraudulent'],
    'Count': [91.1, 8.9]
})

try:
    import plotly.express as px
    fig = px.bar(fig_data, x='Category', y='Count', title="Test Chart")
    st.plotly_chart(fig)
    st.success("✅ Plotly chart working!")
except Exception as e:
    st.error(f"❌ Chart failed: {e}")

st.write("If you see this message, the basic app structure is working!")
