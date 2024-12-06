import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import io

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



# Simulated data (replace with real data in production)
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa'] * 600
    actual_sales = [128.45, 4339.20, 764.80, 1800.00, 239.94] * 600
    predicted_sales = [133.59, 4241.90, 752.40, 1815.50, 230.00] * 600

    # Create DataFrame
    data = pd.DataFrame({
        'Region': regions[:3000],
        'Actual_Sales': actual_sales[:3000],
        'Predicted_Sales': predicted_sales[:3000]
    })

    # Group data by Region
    sales_by_region = data.groupby('Region')[['Actual_Sales', 'Predicted_Sales']].mean().reset_index()

    ### Bar Chart: Actual vs Predicted Sales by Region ###
    st.write("### Bar Chart: Actual vs Predicted Sales by Region")
    bar_fig = px.bar(
        sales_by_region,
        x='Region',
        y=['Actual_Sales', 'Predicted_Sales'],
        title='Actual vs Predicted Sales by Region',
        barmode='group',
        labels={'value': 'Average Sales', 'variable': 'Sales Type'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    ### Histogram: Distribution of Actual vs Predicted Sales ###
    st.write("### Histogram: Distribution of Actual vs Predicted Sales")

    # Create a smaller Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted figure size for a more compact appearance

    # Plot histograms
    ax.hist(data['Actual_Sales'], bins=30, alpha=0.5, label='Actual Sales', color='blue')
    ax.hist(data['Predicted_Sales'], bins=30, alpha=0.5, label='Predicted Sales', color='orange')

    # Set labels, title, and legend
    ax.set_title('Histogram of Actual vs Predicted Sales', fontsize=14)  # Slightly smaller font
    ax.set_xlabel('Sales Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # Display the plot in Streamlit
    st.pyplot(fig)



    ### Scatter Plot: Actual vs Predicted Sales by Region ###
    st.write("### Scatter Plot: Actual vs Predicted Sales by Region")
    scatter_fig = px.scatter(
        data,
        x='Actual_Sales',
        y='Predicted_Sales',
        color='Region',
        title='Scatter Plot of Actual vs Predicted Sales by Region',
        labels={'Actual_Sales': 'Actual Sales', 'Predicted_Sales': 'Predicted Sales'},
        opacity=0.7
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    ### Line Graph: Actual vs Predicted Sales by Region ###
    st.write("### Line Graph: Actual vs Predicted Sales by Region")
    line_fig = px.line(
        sales_by_region,
        x='Region',
        y=['Actual_Sales', 'Predicted_Sales'],
        title='Actual vs Predicted Sales by Region',
        labels={'value': 'Average Sales', 'variable': 'Sales Type'},
        markers=True,
        color_discrete_sequence=['blue', 'orange']
    )
    st.plotly_chart(line_fig, use_container_width=True)

    ### Download Aggregated Data as CSV ###
    st.write("### Download Predicted Sales Data (Aggregated by Region)")
    csv_buffer = io.StringIO()
    sales_by_region.to_csv(csv_buffer, index=False)  # Export grouped data
    st.download_button(
        label="Download Aggregated Sales CSV",
        data=csv_buffer.getvalue(),
        file_name="aggregated_sales_by_region.csv",
        mime="text/csv",
    )

