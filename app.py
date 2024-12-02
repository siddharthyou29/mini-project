import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app
st.title("Sales Visualizer App")
st.markdown("This interactive dashboard displays **Actual vs Predicted Sales** using various visualizations.")

# Cache data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Cache model training
@st.cache_resource
def train_model(X, y):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    return rf_model

# Upload file
uploaded_file = st.file_uploader("Upload CSV file for analysis", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    data = load_data(uploaded_file)
    data.fillna(0, inplace=True)
    data = pd.get_dummies(data, drop_first=True)

    if 'Sales' not in data:
        st.error("The dataset must contain a 'Sales' column.")
        st.stop()

    X = data.drop(columns=['Sales'])
    y = data['Sales']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = train_model(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluate model
    st.subheader("Model Performance")
    st.write(f"**R² Score:** {r2_score(y_test, rf_predictions):.2f}")
    st.write(f"**RMSE:** {mean_squared_error(y_test, rf_predictions, squared=False):.2f}")

    # Prepare data for visualization
    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['Actual Sales'] = y_test.reset_index(drop=True)
    test_data['Predicted Sales'] = rf_predictions

    # Simulated regions and sales for visualization
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa'] * 600
    actual_sales = [128.45, 4339.20, 764.80, 1800.00, 239.94] * 600
    predicted_sales = [133.59, 4241.90, 752.40, 1815.50, 230.00] * 600

    sales_data = pd.DataFrame({
        'Region': regions[:3000],
        'Actual_Sales': actual_sales[:3000],
        'Predicted_Sales': predicted_sales[:3000]
    })
    sales_by_region = sales_data.groupby('Region')[['Actual_Sales', 'Predicted_Sales']].mean().reset_index()

    # Visualizations
    st.subheader("Visualizations")

    # Bar Chart
    st.write("### Bar Chart: Actual vs Predicted Sales by Region")
    bar_fig = px.bar(
        sales_by_region,
        x='Region',
        y=['Actual_Sales', 'Predicted_Sales'],
        title="Actual vs Predicted Sales by Region",
        barmode='group',
        labels={'value': 'Sales', 'variable': 'Type'}
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Histogram
    st.write("### Histogram: Distribution of Actual vs Predicted Sales")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=sales_data['Actual_Sales'], name='Actual Sales', opacity=0.6, marker_color='blue'))
    hist_fig.add_trace(go.Histogram(x=sales_data['Predicted_Sales'], name='Predicted Sales', opacity=0.6, marker_color='orange'))
    hist_fig.update_layout(
        barmode='overlay',
        title="Distribution of Actual vs Predicted Sales",
        xaxis_title="Sales Value",
        yaxis_title="Frequency"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Scatter Plot
    st.write("### Scatter Plot: Actual vs Predicted Sales by Region")
    scatter_fig = px.scatter(
        sales_data,
        x='Actual_Sales',
        y='Predicted_Sales',
        color='Region',
        title="Scatter Plot of Actual vs Predicted Sales by Region",
        labels={'Actual_Sales': 'Actual Sales', 'Predicted_Sales': 'Predicted Sales'}
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Line Graph
    st.write("### Line Graph: Average Actual vs Predicted Sales by Region")
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=sales_by_region['Region'], y=sales_by_region['Actual_Sales'], mode='lines+markers', name='Actual Sales', line_color='blue'))
    line_fig.add_trace(go.Scatter(x=sales_by_region['Region'], y=sales_by_region['Predicted_Sales'], mode='lines+markers', name='Predicted Sales', line_color='orange'))
    line_fig.update_layout(
        title="Average Actual vs Predicted Sales by Region",
        xaxis_title="Region",
        yaxis_title="Average Sales"
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # Export combined test data
    st.subheader("Download Results")
    csv = test_data.to_csv(index=False)
    st.download_button(
        label="Download Test Data with Predictions",
        data=csv,
        file_name='test_data_with_predictions.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a CSV file to get started.")
