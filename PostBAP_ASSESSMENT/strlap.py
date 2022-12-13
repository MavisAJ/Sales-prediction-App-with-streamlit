 # importing librabries
import streamlit as st
import pandas as pd
import numpy as np
import os
import os, pickle
import re
from PIL import Image

st.set_page_config(page_title= "Grocery Store Sales Prediction App", page_icon= ":shopping_bags:", 
layout= "wide", initial_sidebar_state= "auto")

# Setting the page title
st.title("Sales Prediction")
# ---- Importing and creating other key elements items
# Importing machine learning toolkit
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object

# Function to load the dataset
@st.cache()
def load_data(relative_path):
    train_data = pd.read_csv(relative_path, index_col= 0)
    train_data["onpromotion"] = train_data["onpromotion"].apply(int)
    train_data["store_nbr"] = train_data["store_nbr"].apply(int)
    train_data["Sales_date"] = pd.to_datetime(train_data["Sales_date"]).dt.date
    train_data["year"]= pd.to_datetime(train_data['Sales_date']).dt.year
    return train_data

# Function to get date features from the inputs
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df = df.drop(columns = "date")
    return df
# ----- Loading the key components
# Loading the base dataframe
rpath = "/Users/emmanythedon/Documents/train.csv"
train_data= load_data(rpath)

# Loading the toolkit
loaded_toolkit = load_ml_toolkit("/Users/emmanythedon/Documents/PostBAP_ASSESSMENT/ML_items")
if "results" not in st.session_state:
    st.session_state["results"] = []

# Instantiating the elements of the Machine Learning Toolkit
mscaler = loaded_toolkit["scaler"]
ml_model = loaded_toolkit["model"]
encode  = loaded_toolkit["encoder"]
print(type(encode))

# Defining the base containers/ main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()

form = st.form(key="information", clear_on_submit=True)
# Structuring the header section
with header:
    #header.write("Sales Prediction")
    # Icon for the page
    image = Image.open("/Users/emmanythedon/Documents/SALES FORECASTING/sales.jpg")
    st.image(image, width = 500)
# Instantiating the form to receive inputs from the user
st.sidebar.header("This app predicts the sales of the Corporation Favorita grocery store")
check =st.sidebar.checkbox("Click here to know more about your columns")
if check:
    st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **family** identifies the type of product sold.
                    - **sales** is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **sales_date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **cluster** is a grouping of similar stores.
                    - **oil_price** is the daily oil price
                    - **holiday_type** indicates whether the day was a holiday, event day, or a workday
                    - **locale** indicates whether the holiday was local, national or regional.
                    - **transferred** indicates whether the day was a transferred holiday or not.
                    """)
# Structuring the dataset section
with dataset:
       dataset.markdown("**This is the dataset of Corporation Favorita**")
       check = dataset.checkbox("Preview the dataset")
       if check:
            dataset.write(train_data.head())
       dataset.write("View sidebar for information on the columns")
     
# Defining the list of expected variables
expected_inputs = ["Sales_date",  "family",  "store_nbr", "cluster",  "city",  "state",  "onpromotion",  "dcoilwtico ",  "type_y", "locale"]

# List of features to encode
categoricals = ["family", "city", "type_y", "locale"]

# List of features to scale
cols_to_scale = ['dcoilwtico']
with features_and_output:
    features_and_output.subheader("Give us your Inputs")
    features_and_output.write("This section captures your input to be used in predictions")

    col1, col2  = features_and_output.columns(2)

    # Designing the input section of the app

    with form:
        Sales_date = col1.date_input("Select a date:", min_value= train_data["Sales_date"].min())
        family =col1.selectbox("Family of items:",options=(list( train_data['family'].unique())))
        city = col1.selectbox("choose city:",options =(train_data['city'].unique()))
        store_nbr = col1.selectbox("store number:",options =(train_data['store_nbr'].unique()))
        cluster = col1.selectbox("cluster:",options =(train_data['cluster'].unique()))
        dcoilwtico = col1.selectbox("oil prices:",options =(train_data['dcoilwtico'].unique()))
        onpromotion = col2.slider("Select number of items on promo:",min_value =0, max_value = 742,step =1)
        if col2.checkbox("Is it a holiday? (Check if holiday)"):
            print(True)
            type_y = col2.selectbox("Holiday type:", options=(train_data["type_y"].unique()))
            locale = col2.selectbox("Locale:", options=(train_data["locale"].unique()))
            transferred = col2.radio('Was the holiday transferred',
            ('True','False'))
            #print((train_data["type_y"].unique()))
        else:
            type_y = "Work Day"
            locale = "National"
            transferred = 'False'
        # Submit button
        submitted = form.form_submit_button(label= "Submit")
        transactions = col2.slider("Select the number transactions for this day:",min_value =0, max_value = 8359,step = 100)
    if submitted:
        st.success('All Done!', icon="✅")  
        # Inputs formatting
        input_dict = {
            "Sales_date": [Sales_date],
            "family": [family],
            "store_nbr": [store_nbr],
            "cluster": [cluster],
            "city": [city],
            "onpromotion": [onpromotion],
            "dcoilwtico": [dcoilwtico],
            "type_y": [type_y],
            "locale": [locale],
            "transferred":[transferred],
            "transactions" :[transactions]
        }

        # Converting the input into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_df = input_data.copy()
        
        # Converting data types into required types
        input_data["Sales_date"] = pd.to_datetime(input_data["Sales_date"]).dt.date
        input_data[cols_to_scale] = input_data[cols_to_scale].apply(int)
        
        # Getting date features
        df_processed = getDateFeatures(input_data, "Sales_date")

        df_processed['year'] = pd.to_datetime(df_processed['Sales_date']).dt.year
        df_processed['month'] = pd.to_datetime(df_processed['Sales_date']).dt.month
        df_processed['week'] = pd.to_datetime(df_processed['Sales_date']).dt.week
        df_processed['day'] = pd.to_datetime(df_processed['Sales_date']).dt.day
        df_processed.drop(columns=["Sales_date"], inplace= True)
        # Scaling the columns
        df_processed[cols_to_scale] = mscaler.transform(df_processed[cols_to_scale])

        # Encoding the categoricals
        print(categoricals)
        print(encode.feature_names_in_)
        #print(encoded_categoricals.get_feature_names_out().tolist())
        
        encoded_categoricals = encode.transform(input_data[categoricals])
        encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = encode.get_feature_names_out().tolist())
        df_processed = df_processed.join(encoded_categoricals)
        df_processed.drop(columns=categoricals, inplace=True)
        df_processed.rename(columns= lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace= True)
        df_processed["transferred"] = np.where(df_processed["transferred"] == "True", 0, 1)
        # Making the predictions        
        dt_pred = ml_model.predict(df_processed)
        df_processed["sales"] = dt_pred
        input_df["sales"] = dt_pred
        display = dt_pred[0]

        # Adding the predictions to previous predictions
        st.session_state["results"].append(input_df)
        result = pd.concat(st.session_state["results"])


        # Displaying prediction results
        st.success(f"**Predicted sales**: USD {display}")

       # Expander to display previous predictions
        previous_output = st.expander("**Review previous predictions**")
        previous_output.dataframe(result, use_container_width= True)
    
    
# ----- Defining and structuring the footer
footer = st.expander("**Additional Information**")
with footer:
    footer.markdown("""
                    - You may access the repository in which the model was built [here](https://github.com/MavisAJ/Store-sales-prediction-Regression___TimeSeries-Analysis-.git).
                    - This is my first attempt at a Streamlit project so I would love to hear your criticisms.
                    - You may also connect with me [here](https://kodoi-oj.github.io/).
                    - *KME*
                    """)
