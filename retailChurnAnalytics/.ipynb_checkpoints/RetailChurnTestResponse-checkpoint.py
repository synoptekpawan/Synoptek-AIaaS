import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
import datetime as dt
from datetime import timedelta, datetime
np.random.seed(10)
random.seed(10)
import sys
sys.path.insert(0, r"./retailChurnAnalytics/")
from evaluateModelOnHoldData import evalModel


holdOuts = r"./retailChurnAnalytics/holdOutData/"
outputs = r"./retailChurnAnalytics/outputs/"
models = r"./retailChurnAnalytics/models/"
logs = r"./retailChurnAnalytics/logs/"

# -----------------------------------------------------------------------------------
## prepare logging config
logging.basicConfig(filename=logs+'modelEvalMain.log', 
                    level = logging.DEBUG,
                    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

# -----------------------------------------------------------------------------------
def RetailChurnTestResponse (): #holdOuts, outputs, models
    try:
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**User Data**')
            userData = pd.DataFrame()
            UserId = st.text_input("User Id")
            if len(UserId) > 0:
                userData['UserId'] = [UserId]

            # Add a slider to the sidebar
            Age = st.slider("Age", 0, 100, 0)

            if Age > 0:
                userData['Age'] = [Age]

            if st.checkbox("Gender", key='7'):
                options = ["F", "M"]

                # Create a radio button
                Gender = st.radio("Select Gender",options)

                # Display the selected option
                #st.write("Gender: ", Gender)
                userData['Gender'] = [Gender]

            if st.checkbox("Address", key='8'):
                options = ["AL", "CA", "DC", "FL", "HI", "ME"]

                # Create a radio button
                Address = st.radio("Select Address",options)

                # Display the selected option
                #st.write("Address: ", Address)
                userData['Address'] = [Address]

            if st.checkbox("User Type", key='9'):
                options = ["Fast", "Inspiration", "Value"]

                # Create a radio button
                userType = st.radio("Select User Type", options)

                # Display the selected option
                #st.write("User Type: ", userType)
                userData['userType'] = [userType]

            if len(userData)> 0:
                st.dataframe(userData)

        with col2:
            st.markdown('**Activity Data**')
            activityData = pd.DataFrame()
            # Add a slider to the sidebar   
            TransactionId = st.text_input("Transaction Id")
            if len(TransactionId) > 0:
                activityData['TransactionId'] = [TransactionId]

            today_ = dt.datetime.today().date()
            dt_ = st.date_input("Enter Date", today_, label_visibility = 'hidden')
            if len(str(dt_)) > 0:
                activityData['Timestamp'] = [dt_]

            if len(UserId) > 0:
                activityData['UserId'] = [UserId]

            ItemId = st.text_input("Item Id")
            if len(ItemId) > 0:
                activityData['ItemId'] = [ItemId]

            # Add a slider to the sidebar
            Quantity = st.slider("Quantity", 0, 50, 0)

            if Quantity > 0:
                activityData['Quantity'] = [Quantity]

            # Add a slider to the sidebar
            Value = st.text_input("Value")
            if len(Value) > 0:
                activityData['Value'] = [Value]

            # Add a slider to the sidebar
            Location = st.text_input("Location")
            if len(Location) > 0:
                activityData['Location'] = [Location]

            if st.checkbox("Product Category", key='10'):
                options = ["Apparels","Books","Consumer Electronics","Fashion Accessories",
                           "Food","Footwear","Health & Beauty Supplements","Home Decor",
                           "Jwellary","Kitchen & Home Appliances","Mobile Phones","Toys & Games"]

                # Create a radio button
                ProductCategory = st.radio("Select Product Category", options)

                # Display the selected option
                #st.write("User Type: ", userType)
                activityData['ProductCategory'] = [ProductCategory]

            if len(activityData)> 0:
                st.dataframe(activityData)
                
        if (len(userData) > 0) & (len(activityData) > 0):
            userData.to_csv(holdOuts+"userDataHdo.csv", index=False)
            activityData.to_csv(holdOuts+"activityDataHdo.csv", index=False)
            
            st.button("Run Model", key='20')
                
            churnPredDf = evalModel(userData, activityData, holdOuts, outputs, models)

            st.dataframe(churnPredDf)

            st.success('Thanks for using the service')

    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__RetailChurnTestResponse__':
    RetailChurnTestResponse()

