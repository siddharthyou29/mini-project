import streamlit as st
import pandas as pd
st.write('Hello World')
x=st.text_input('Enter text:')
st.write(f'Your text is: {x}')

data = pd.read_csv('Sales_Profit_Data_Electrical_Appliances.csv')
st.write(data)