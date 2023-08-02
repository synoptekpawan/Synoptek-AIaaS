import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pkl file 
reg_avg_tkt = joblib.load('C:/Users/Dishita Neve/Sales Forecast/Model/mod_avg_tkt.pkl')
reg_ord = joblib.load('C:/Users/Dishita Neve/Sales Forecast/Model/mod_ord.pkl')

# Reading the original csv file
df = pd.read_csv('C:/Users/Dishita Neve/Sales Forecast/Input/Champion_Data_Historical_Data.csv')
df['Month'] = pd.to_datetime(df['Month'])
# Adding Year and Month Column
df['Year'] = df['Month'].dt.year
df['month'] = df['Month'].dt.month
df.dropna(subset=['Budget'], inplace=True)
df = df.fillna(0)

# Set the title of the web app
st.title("Sales Forecast")

# Ask user to enter the month
month1 = st.number_input("**Select Forecast Period (Months):**")

# Select location from available options
location_options = ['ALBANY','ALBUQUERQUE','ATLANTA','AUSTIN','BALTIMORE','BIRMINGHAM','BOISE','CHARLOTTE','CHATTANOOGA','CHICAGO','CINCINNATI','CLEVELAND','COLORADOSP','COLUMBIA','COLUMBUSOH','DAYTON','DENVER','EVANSVILLE','FTCOLLINS','FTWORTH','GRANDRAPIDS','GREENSBORO','HUNTSVILLE','INDIANAPOLIS','JOHNSONCITY','KANSASCITY','LEXINGTON','LOUISVILLE','MACON','MEMPHIS','MILWAUKEE','MINNEAPOLIS','NASHVILLE','OKLAHOMACITY','OMAHA','PITTSBURGH','PORTLAND','RALEIGH','RICHMONDVA','ROCHESTER','SALTLAKECITY','SANANTONIO','SEATTLE','SOUTHBEND','STLOUIS','TOLEDO','TULSA','WICHITA']
selected_location = st.selectbox("**Select a location from markdown**", location_options)

my_dict = {'ALBANY': 0.0, 'ALBUQUERQUE': 1.0, 'ATLANTA': 2.0, 'AUSTIN': 3.0, 'BALTIMORE': 4.0, 'BIRMINGHAM': 5.0, 'BOISE': 6.0, 'CHARLOTTE': 7.0, 'CHATTANOOGA': 8.0, 'CHICAGO': 9.0, 'CINCINNATI': 10.0, 'CLEVELAND': 11.0, 'COLORADOSP': 12.0, 'COLUMBIA': 13.0, 'COLUMBUSOH': 14.0, 'DAYTON': 15.0, 'DENVER': 16.0, 'EVANSVILLE': 17.0, 'FTCOLLINS': 18.0, 'FTWORTH': 19.0, 'GRANDRAPIDS': 20.0, 'GREENSBORO': 21.0, 'HUNTSVILLE': 22.0, 'INDIANAPOLIS': 23.0, 'JOHNSONCITY': 24.0, 'KANSASCITY': 25.0, 'LEXINGTON': 26.0, 'LOUISVILLE': 27.0, 'MACON': 28.0, 'MEMPHIS': 29.0, 'MILWAUKEE': 30.0, 'MINNEAPOLIS': 31.0, 'NASHVILLE': 32.0, 'OKLAHOMACITY': 33.0, 'OMAHA': 34.0, 'PITTSBURGH': 35.0, 'PORTLAND': 36.0, 'RALEIGH': 37.0, 'RICHMONDVA': 38.0, 'ROCHESTER': 39.0, 'SALTLAKECITY': 40.0, 'SANANTONIO': 41.0, 'SEATTLE': 42.0, 'SOUTHBEND': 43.0, 'STLOUIS': 44.0, 'TOLEDO': 45.0, 'TULSA': 46.0, 'WICHITA': 47.0}
sel_loc = my_dict[selected_location]

df1 = df[df['Location'] == selected_location]
year = df1['Year'].max()
df2 = df1[df1['Year'] == year]
month = df2['month'].max()
m = [i + 1  for i in range(int(month1))]
y = [year for i in range(int(month1))]
for j in m : 
    if j > 12 :
        year = year + 1
        k = m.index(j)
        m1 = m[k:]
        y1 = y[:k]
        m3 = m[:k]
        m2 = [n + 1 for n in range(len(m1))]
        y2 = [y[0] +1 for n in range(len(m1))]
        m = m3 + m2
        y = y1 + y2
        break

# Ask user to enter the budget
budget = st.number_input("**Enter the budget**")
    
def assign_seasonality(order):
    if order in [11,12,1,2]:
        return 1
    elif order in [3,4,5,6]:
        return 2
    elif order in [7,8,9,10] :
        return 3
    
# Button to trigger the sales forecast
if st.button("Select"):
    if budget < 0:
        st.error("Invalid budget! Please enter a non-negative value.")
    else:
        input_data = pd.DataFrame({
            'Location': [sel_loc]*int(month1),
            'Budget': [budget]*int(month1),
            'Year' : y,
            'month': m                              
        })

        ##input_data['category'] = input_data['Orders'].apply(assign_category)
        input_data['seasonality'] = input_data['month'].apply(assign_seasonality)

        input_data['month'] = input_data['month'].astype(int)
        input_data['Budget'] = input_data['Budget'].astype(int)
        input_data['Year'] = input_data['Year'].astype(int)
        input_data['Location'] = input_data['Location'].astype(int)
        input_data['seasonality'] = input_data['seasonality'].astype(int)

        # Make the forecast for Average.ticket
        avg_ticket = reg_avg_tkt.predict(input_data[['Location','Year','month','seasonality','Budget']])
        input_data['Average.ticket'] = np.round(abs(avg_ticket))

        ord = reg_ord.predict(input_data[['Location','Budget','Year','month','seasonality','Average.ticket']])
        input_data['Orders'] = np.round(abs(ord))
    
        sales_forecast = np.round(np.round(abs(ord))*avg_ticket)
        input_data['Sales Forecast'] = abs(sales_forecast)

        if (budget == 0) :
            input_data['Sales Forecast'] = 0
            input_data['avg_ticket'] = 0
            input_data['Orders'] = 0

        input_data = input_data[['Year','month','Average.ticket','Orders','Sales Forecast']]
        st.table(input_data)

        # Display the sales forecast to the user
        #st.write("**Sales Forecast for the above input would be :**", abs(sales_forecast[0]))
        #st.write("**Average Ticket for the above input would be :**", abs(input_data['Average.ticket'][0]))
        #st.write("**Number of orders for the above input would be :**",abs(input_data['Orders'][0]))
        #st.write("**Conversion ran for the above input would be :**", abs(input_data['Conversion.ran'][0]))
        #st.write("**Conversion Close for the above input would be :**", abs(input_data['Conversion.close'][0]))
        #st.write("**APPT ran for the above input would be :**", abs(input_data['APPT.ran'][0]))
        #st.write("**APPT Created for the above input would be :**", abs(input_data['APPT.created'][0]))

        df1 = df[df['Location'] == selected_location]
        st.info("**The minumum and maximum sales for the location** "+ str(selected_location) + " **is between** " + str(df['Sales'].min()) + " **and** "+ str(df['Sales'].max()))


