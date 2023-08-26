import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from neuralprophet import NeuralProphet


#insert custom styling with CSS
st.markdown('<link rel="stylesheet" href="styles.css">', unsafe_allow_html=True)

# Abre el archivo comprimido
data = pd.read_csv('https://github.com/OmdenaAI/kansas-chapter-crime-rates/blob/main/src/Kansas_Crime_rate_app/data_kansas_EDA.csv.gz', compression='gzip')

#image in the sidebar
image = Image.open('kansas_flag.png')
st.sidebar.image(image, caption='Ad astra per aspera')

#title of the app
st.sidebar.title(':blue[Kansas Crime analysis by year]')

#widgets of the sidebar of the app
year= st.sidebar.slider('SELECT YEAR:', 2016, 2020)

##### FORECASTING #####

#widget for the forecasting
#loading the data
@st.cache_data
def forecasting_data():
    df = pd.read_csv('crime_data_total_modelling.csv')
    return df

df= forecasting_data()

crime_type= ['Murder', 'Rape', 'Robbery', 'Assault', 'Burglary', 'Theft', 'Motor_Vehicle_Theft', 'Arson']

st.sidebar.title('Forecasting')
type_of_crime= st.sidebar.selectbox('SELECT TYPE OF CRIME', crime_type)


st.sidebar.markdown('''
---
Created by Elianneth Cabrera
''')

##### VISUALIZATIONS ####

#analysis of the victims
#filtered the data
data_year = data[data['DATA_YEAR'] == year]

col1, col2 = st.columns(2)

with col1:
    st.header("Age of the victims by sex")
    fig = px.histogram(data_year, x='AGE_NUM_VICT', color='SEX_CODE_VICT', barmode='group', nbins=10)
    fig.update_xaxes(title_text='Victim age') 
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')

with col2:
    st.header("Race of the victims by sex")
    fig = px.histogram(data_year, x='RACE_ID_VICT', color='SEX_CODE_VICT', barmode='group')
    fig.update_xaxes(title_text='Victim race') 
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')

#analysis of the offender
col3, col4 = st.columns(2)

with col3:
    st.header("Age of the offender by sex")
    fig = px.histogram(data_year, x='AGE_NUM', color='SEX_CODE', barmode='group', nbins=10)
    fig.update_xaxes(title_text='Offender age') 
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')

with col4:
    st.header("Race of the offender by sex")
    fig = px.histogram(data_year, x='RACE_ID', color='SEX_CODE', barmode='group')
    fig.update_xaxes(title_text='Offender race') 
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')


#type of offenses and weapons
col5, col6 = st.columns(2)

with col5:
    st.header("Incidents by day of the week")

    day_order = ['Mon', 'Tue', 'Wed', 'Thue', 'Fri', 'Sat', 'Sun']
    data_year['DAY_OF_WEEK'] = pd.Categorical(data_year['DAY_OF_WEEK'], categories=day_order, ordered=True)

    #group by year en day of the week
    day_of_incident= data_year.groupby(['DATA_YEAR', 'DAY_OF_WEEK'])['INCIDENT_ID'].count()
    day_of_incident= day_of_incident.to_frame().reset_index()

    #creating a pivot table
    day_of_week= day_of_incident.pivot(index= 'DAY_OF_WEEK', columns='DATA_YEAR', values= 'INCIDENT_ID')

    fig = px.line(day_of_week)
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')
   
with col6:
    st.header("Weapons used by offenders")
    filtered_df_weapon = data_year[data_year['WEAPON_ID'] != 'Unarmed']
    fig = px.bar(filtered_df_weapon['WEAPON_ID'].value_counts(),
                orientation='h', labels={'index': 'Type of Weapon', 'value': 'Count'})
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')

#most commons crimes against person and property
col7, col8 = st.columns(2)

with col7:
    st.header("5 most common crimes against Persons")

    #filtred the data for crime against persons
    crime_person= data_year[data_year['CRIME_AGAINST']== 'Person']

    #Get the 10 most frequent types of offenses
    top_offense_person = crime_person['OFFENSE_TYPE_ID'].value_counts().head(5).index

    #Filter the DataFrame by the most frequent offenses
    df_top_offense_person = crime_person[crime_person['OFFENSE_TYPE_ID'].isin(top_offense_person)]

    #Create the horizontal bar chart with Plotly Express
    fig = px.bar(df_top_offense_person['OFFENSE_TYPE_ID'].value_counts(),
                orientation='h', labels={'index': 'Type of offense', 'value': 'Count'})
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')
   
with col8:
    st.header("5 most common offense against Property")

    #filtred the data for crime against persons
    crime_property= data_year[data_year['CRIME_AGAINST']== 'Property']

    #Get the 10 most frequent types of offenses
    top_offense_property = crime_property['OFFENSE_TYPE_ID'].value_counts().head(5).index

    #Filter the DataFrame by the most frequent offenses
    df_top_offense_property = crime_property[crime_property['OFFENSE_TYPE_ID'].isin(top_offense_property)]

    #Create the horizontal bar chart with Plotly Express
    fig = px.bar(df_top_offense_property['OFFENSE_TYPE_ID'].value_counts(),
                orientation='h', labels={'index': 'Type of offense', 'value': 'Count'})
    st.plotly_chart(fig, use_container_width=True, key='styled-graph')

##Forecasting chart##

#filtered the data for the forecasting
df['Year']= pd.to_datetime(df['Year'], format='%Y')
df_forecasting= df[['Year', type_of_crime]]
df_forecasting.columns= ['ds', 'y']

# Initialize the NeuralProphet model
model = NeuralProphet()

# Fit the model to the training data
model.fit(df_forecasting)

# Generate forecasts for the testing period
future = model.make_future_dataframe(df_forecasting, periods= 5, n_historic_predictions=len(df_forecasting))
forecast = model.predict(future)

#ploting the forecast
plot_1 = model.plot(forecast)
st.header(f"Forecasting for {type_of_crime}")
st.plotly_chart(plot_1, use_container_width=True, key='styled-graph')



