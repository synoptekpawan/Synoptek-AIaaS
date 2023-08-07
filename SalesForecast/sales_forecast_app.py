# Importing Libraries
from dateutil.relativedelta import relativedelta
from azure.storage.blob import BlobServiceClient
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

storage_account_key = os.environ['storage_account_key']
storage_account_name = os.environ['storage_account_name']
connection_string = os.environ['connection_string']
container_name = os.environ['container_name']

# Load the pkl file 
p = r"./Sales_Forecast/Model/"
#reg_avg_tkt = joblib.load('C:/Users/dneve/Sales_Forecast/Model/mod_avg_tkt.pkl')
reg_avg_tkt = joblib.load(p+'mod_avg_tkt.pkl')
#reg_ord = joblib.load('C:/Users/dneve/Sales_Forecast/Model/mod_ord.pkl')
reg_ord = joblib.load(p+'mod_ord.pkl')
#reg_appt_crt = joblib.load('C:/Users/dneve/Sales_Forecast/Model/mod_appt_created.pkl')
reg_appt_crt = joblib.load(p+'mod_appt_created.pkl')

# Reading the original csv file
inp = r"./Input/"
df = pd.read_excel(inp + 'Data_Cleaning.xlsx')

def salesforecast(reg_avg_tkt,reg_ord,reg_appt_crt,df) :
    # Set the title of the web app
    st.title("Sales Forecast")

    forecasted_months = st.number_input("**Select Forecast Period (Months):**")

    # Select location from available options
    location_options = ['Albany', 'Albuquerque', 'Atlanta', 'Austin', 'Baltimore', 'Birmingham', 'Boise', 'Charlotte', 'Chattanooga', 'Chicago', 'Cincinnati', 'Cleveland', 'Coloradosp', 'Columbia', 'Columbusoh', 'Dayton', 'Denver', 'Evansville', 'Ftcollins', 'Ftworth', 'Grandrapids', 'Greensboro', 'Huntsville', 'Indianapolis', 'Johnsoncity', 'Kansascity', 'Lexington', 'Louisville', 'Macon', 'Memphis', 'Milwaukee', 'Minneapolis', 'Nashville', 'Oklahomacity', 'Omaha', 'Pittsburgh', 'Portland', 'Raleigh', 'Richmondva', 'Rochester', 'Saltlakecity', 'Sanantonio', 'Seattle', 'Southbend', 'Stlouis', 'Toledo', 'Tulsa', 'Wichita']
    selected_location = st.selectbox("**Select a location from markdown**", location_options)

    my_dict = {'Albany': 0.0, 'Albuquerque': 1.0, 'Atlanta': 2.0, 'Austin': 3.0, 'Baltimore': 4.0, 'Birmingham': 5.0, 'Boise': 6.0, 'Charlotte': 7.0, 'Chattanooga': 8.0, 'Chicago': 9.0, 'Cincinnati': 10.0, 'Cleveland': 11.0, 'Coloradosp': 12.0, 'Columbia': 13.0, 'Columbusoh': 14.0, 'Dayton': 15.0, 'Denver': 16.0, 'Evansville': 17.0, 'Ftcollins': 18.0, 'Ftworth': 19.0, 'Grandrapids': 20.0, 'Greensboro': 21.0, 'Huntsville': 22.0, 'Indianapolis': 23.0, 'Johnsoncity': 24.0, 'Kansascity': 25.0, 'Lexington': 26.0, 'Louisville': 27.0, 'Macon': 28.0, 'Memphis': 29.0, 'Milwaukee': 30.0, 'Minneapolis': 31.0, 'Nashville': 32.0, 'Oklahomacity': 33.0, 'Omaha': 34.0, 'Pittsburgh': 35.0, 'Portland': 36.0, 'Raleigh': 37.0, 'Richmondva': 38.0, 'Rochester': 39.0, 'Saltlakecity': 40.0, 'Sanantonio': 41.0, 'Seattle': 42.0, 'Southbend': 43.0, 'Stlouis': 44.0, 'Toledo': 45.0, 'Tulsa': 46.0, 'Wichita': 47.0}
    sel_loc = my_dict[selected_location]

    # Ask user to enter the budget
    budget1 = st.text_input("**Enter the budget**")


    # Adding seasonlity parameter    
    def assign_seasonality(order):
        if order in [11,12,1,2]:
            return 1
        elif order in [3,4,5,6]:
            return 2
        elif order in [7,8,9,10] :
            return 3
        
    # Checking negative values in budget    
    def check_list_elements(lst):
        for element in lst:
            if element < 0:
                raise ValueError("Negative value found in the list")
            
    # Uploading prediction file in Azure Blob Storage
    def uploadToBlobStorage(file_path,file_name):
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        if blob_client.exists():
            # Delete the existing blob
            blob_client.delete_blob()
        with open(file_path,"rb") as data:
            blob_client.upload_blob(data)
            print(f"Uploaded {file_name}.")
            
    # Button to trigger the sales forecast
    if st.button("Select"):

        #for bval in budget :
        try :
            forecasted_months = int(forecasted_months)
            result = 10 / forecasted_months
        except ZeroDivisionError:
            st.write("You cannot divide by zero.")
        except ValueError:
            st.write("Invalid input. Please enter a valid number.")

        else : 
            try :
                budget = budget1.split()
                budget = budget[:int(forecasted_months)]
                budget = [int(element) for element in budget]
                check_list_elements(budget)
            except ValueError as e:
                st.error("Invalid budget! Please enter a non-negative value.")
            else:
                
                df1 = df[df['City'] == selected_location]
                year = df1['Year'].max()
                df2 = df1[df1['Year'] == year]
                month = df2['Month'].max()

                # Find the maximum date value in the 'Date' column
                df1['Date'] = pd.to_datetime(df1['Date'])
                max_date = df1['Date'].max()

                # Add the specified number of months to the maximum date value
                c = []
                for i in range(1,int(forecasted_months)+1) :
                    forecasted_date = max_date + relativedelta(months=i)
                    c.append(forecasted_date)

                # Create a new DataFrame with the forecasted date
                forecast_df = pd.DataFrame({'Date': c})
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                # Extract the year and month into separate columns
                forecast_df['Year'] = forecast_df['Date'].dt.year
                forecast_df['Month'] = forecast_df['Date'].dt.month


                input_data = pd.DataFrame({
                    'Location_n': [sel_loc]*int(forecasted_months),
                    'City' : selected_location,
                    'Budget': budget,
                    'Year' : forecast_df['Year'].tolist(),
                    'Month': forecast_df['Month'].tolist(),  
                    'Horizon' : [forecasted_months]*int(forecasted_months)      
                })

                ##input_data['category'] = input_data['Orders'].apply(assign_category)
                input_data['seasonality'] = input_data['Month'].apply(assign_seasonality)

                input_data['Month'] = input_data['Month'].astype(int)
                input_data['Budget'] = input_data['Budget'].astype(int)
                input_data['Year'] = input_data['Year'].astype(int)
                input_data['Location_n'] = input_data['Location_n'].astype(int)
                input_data['seasonality'] = input_data['seasonality'].astype(int)
                input_data['Date'] = input_data['Year'].astype(str) + '-'  + input_data['Month'].astype(str) + '-1' 

                # Make the forecast for APPT.Created
                appt_crt = reg_appt_crt.predict(input_data[['Location_n','Budget','Year','Month','seasonality']])
                input_data['APPT.created'] = np.round(abs(appt_crt))

                # Make the forecast for Average.ticket
                avg_ticket = reg_avg_tkt.predict(input_data[['Location_n','Year','Month','seasonality','Budget']])
                input_data['Average.ticket'] = np.round(abs(avg_ticket))

                input_data['CPA'] = input_data['Budget']/input_data['APPT.created']
                input_data['CPA'] = input_data['CPA'].round()
                print('Appt created',input_data['APPT.created'])

                # If CPA is NaN which means budget is 0 so solving this
                input_data.loc[input_data["Budget"] == 0, ["CPA"]] = 0

                # Make the forecast for Orders
                ord = reg_ord.predict(input_data[['Location_n','Budget','Year','Month','seasonality','Average.ticket','CPA']])
                input_data['Pred Orders'] = np.round(abs(ord))
            
                sales_forecast = np.round(np.round(abs(ord))*avg_ticket)
                input_data['Forecast Sales'] = abs(sales_forecast)

                input_data['Forecast Sales'] = input_data['Forecast Sales'].astype(int)
                input_data['Pred Orders'] = input_data['Pred Orders'].astype(int)
                input_data['Average.ticket'] = input_data['Average.ticket'].astype(int)
                
                # Update columns based on condition
                input_data.loc[input_data["Budget"] == 0, ["Forecast Sales", "Average.ticket","Pred Orders","CPA"]] = 0

                input_data1 = input_data[['Date','Year','Month','City','APPT.created','Budget','CPA','Average.ticket','Pred Orders','Forecast Sales','Horizon']]
                st.table(input_data1)

                df1 = df[df['City'] == selected_location]
                st.info("**The minumum and maximum sales for the location** "+ str(selected_location) + " **is between** " + str(df['Sales'].min()) + " **and** "+ str(df['Sales'].max()))
                
                input_data1 = input_data1.rename(columns={'CPA': 'Pred CPA'})
                input_data1 = input_data1.rename(columns={'Average.ticket': 'Pred Average.ticket'})
                input_data1 = input_data1.rename(columns={'APPT.created': 'Pred APPT.created'})
                input_data1 = input_data1.rename(columns={'Budget': 'Pred Budget'})
                
                df1 = df1.rename(columns={'Sales': 'Actual Sales'})
                df1 = df1.rename(columns={'Orders': 'Act Orders'})
                df1 = df1.rename(columns={'CPA': 'Act CPA'})
                df1 = df1.rename(columns={'Average.ticket': 'Act Average.ticket'})
                df1 = df1.rename(columns={'APPT.created': 'Act APPT.created'})
                df1 = df1.rename(columns={'Budget': 'Act Budget'})
                df1['Horizon'] = forecasted_months
                df1 = df1[['Date','Year','Month','City','Act APPT.created','Act Budget','Act CPA','Act Average.ticket','Act Orders','Actual Sales','Horizon']]
                concatenated = pd.concat([df1, input_data1], ignore_index=True)
                # Storing Streamlit file in local
                concatenated.to_excel('C:/Users/dneve/Sales_Forecast/Input/Streamlit_Output.xlsx',index=False)

                # Uploading Streamlit Output file to Azure Blob Storage from local
                PATH_OF_FILE_TO_UPLOAD = 'C:/Users/dneve/Sales_Forecast/Input/Streamlit_Output.xlsx'
                FILE_NAME = 'Streamlit_Output.xlsx'
                uploadToBlobStorage(PATH_OF_FILE_TO_UPLOAD,FILE_NAME)


    options = st.selectbox("**Select a location from markdown**", [None,'Data Visualization Report','Prediction Report'])

    if options == 'Data Visualization Report' :
        st.markdown("""
            <iframe title="Sales_Forecast_Data_Visualization" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=581032bf-3f5c-475c-b3b1-48f64a844bf9&autoAuth=true&ctid=150d7f46-ce04-4895-b4bd-7cf3fdf622fa" frameborder="0" allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)

    if options == 'Prediction Report' :           
        st.markdown("""
            <iframe title="Sales_Forecast_Azure_Blob_Storage" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=ea0b7da0-21c1-4a8e-947d-d59d02ca1501&autoAuth=true&ctid=150d7f46-ce04-4895-b4bd-7cf3fdf622fa" frameborder="0" allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)

if __name__ == '__salesforecast__':
    salesforecast(reg_avg_tkt,reg_ord,reg_appt_crt,df)
