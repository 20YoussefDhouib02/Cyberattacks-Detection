import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import MinMaxScaler

st.title('Cyberattacks Detector')

with st.sidebar:
    st.header('Data requirements')
    st.caption('Make sure to provide the CSV version of the data')
    with st.expander('Data format'):
        st.markdown(' - Utf-8')
        st.markdown(' - Separated by coma')
        st.markdown(' - First row - header')
        st.markdown(' - The following columns must be included in your data (in any order): pkSeqID, proto, saddr, sport, daddr, dport, seq, mean, stddev, min, max, srate, drate, N_IN_Conn_P_DstIP, N_IN_Conn_P_SrcIP.')       
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by YD</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        st.header('Uploaded Data Sample')
        st.write(df.head(10))
        columns_to_keep = ['pkSeqID','proto','saddr','sport','daddr','dport','seq','mean','stddev','min','max','srate','drate','N_IN_Conn_P_DstIP','N_IN_Conn_P_SrcIP']
        df = df[columns_to_keep]
        df = df.dropna()
        st.header('Cleaned Data Sample Version')
        st.write(df.head(10))

        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()  
        for col in categorical_cols:
            if df[col].apply(type).nunique() > 1:  
                df[col] = df[col].astype(str)  
            df[col] = le.fit_transform(df[col]) 

        scaler = MinMaxScaler()  
        df_scaled = scaler.fit_transform(df)

        df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns)  
        st.header('Prepared Data Sample Version')
        st.write(df_scaled_df.head(10))  
        
        category_model = joblib.load('category_model.joblib')
        subcategory_model = joblib.load('subcategory_model.joblib')

        category_predicted=category_model.predict(df_scaled)
        subcategory_predicted=subcategory_model.predict(df_scaled)

        pred = pd.DataFrame({
            'Category': category_predicted,
            'Subcategory': subcategory_predicted
        })

        # Create the new column based on the conditions
        pred['Attack'] = pred.apply(lambda row: '0' if row['Category'] == 'Normal' and row['Subcategory'] == 'Normal' else '1', axis=1)

        # Reorder the columns to place 'Attack' first
        pred = pred[['Attack', 'Category', 'Subcategory']]

        st.header('Predicted Values')
        st.write(pred.head(10))

        pred = pred.to_csv(index=False).encode('utf-8')
        st.download_button('Download predictions file',
                        pred,
                        'Cyberattacks Predictions.csv',
                        'text/csv',
                        key='download-csv')

