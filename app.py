import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Sales Prediction Dashboard")

# App header
st.title("Sales Prediction Dashboard")
st.subheader("Upload your dataset to get started")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

@st.cache_data
def load_and_process_data(file):
    # Load the dataset and process it
    data = pd.read_csv(file)
    data.fillna(0, inplace=True)  # Handle missing values
    raw_data = data.copy()  # Save raw dataset for display
    data = pd.get_dummies(data, drop_first=True)  # Encode categorical data
    return raw_data, data

@st.cache_data
def train_model(X_train, y_train):
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if uploaded_file:
    # Load and preprocess data
    raw_data, processed_data = load_and_process_data(uploaded_file)

    # Display the raw dataset
    st.write("### Dataset Preview")
    st.dataframe(raw_data)

    # Define features (X) and target (y)
    target_column = 'Sales'
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    rf_model = train_model(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, rf_predictions)
    rmse = mean_squared_error(y_test, rf_predictions, squared=False)

    # Model performance
    st.subheader("Model Performance")
    st.write(f"**Random Forest R² Score:** {r2:.4f}")
    st.write(f"**Random Forest RMSE:** {rmse:.2f}")

    # Simulated regions and predicted sales (replace with real data for better visualization)
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa'] * 600
    predicted_sales = [133.59, 4241.90, 752.40, 1815.50, 230.00] * 600
    data_viz = pd.DataFrame({
        'Region': regions[:3000],
        'Predicted_Sales': predicted_sales[:3000]
    })

    # Bar Chart: Predicted Sales by Region
    st.write("### Predicted Sales by Region (Bar Chart)")
    sales_by_region = data_viz.groupby('Region')['Predicted_Sales'].mean().reset_index()
    bar_fig = px.bar(
        sales_by_region,
        x='Region',
        y='Predicted_Sales',
        title='Predicted Sales by Region',
        labels={'Predicted_Sales': 'Average Predicted Sales'},
        color='Region',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Histogram: Predicted Sales
    st.write("### Histogram of Predicted Sales")
    hist_fig = px.histogram(
        data_viz,
        x='Predicted_Sales',
        nbins=30,
        title='Distribution of Predicted Sales',
        labels={'Predicted_Sales': 'Sales Value'},
        color_discrete_sequence=['#FFA726'],
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Scatter Plot: Predicted Sales by Region
    st.write("### Scatter Plot of Predicted Sales by Region")
    scatter_fig = px.scatter(
        data_viz,
        x='Region',
        y='Predicted_Sales',
        title='Scatter Plot of Predicted Sales by Region',
        color='Region',
        labels={'Predicted_Sales': 'Sales Value'},
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Line Graph: Predicted Sales by Region
    st.write("### Line Graph of Predicted Sales by Region")
    line_fig = px.line(
        sales_by_region,
        x='Region',
        y='Predicted_Sales',
        title='Trend of Predicted Sales by Region',
        labels={'Predicted_Sales': 'Average Predicted Sales'},
        markers=True,
        color_discrete_sequence=['#FF7043'],
    )
    st.plotly_chart(line_fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to start.")
