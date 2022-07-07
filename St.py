import pandas as pd
import streamlit as st 
cost = pd.read_excel("C:\Data science\8888\Project Dataset\cost_and_price.xlsx")
st.write(cost.head())

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

# st.cache 
def get_data(filename):
	cost = pd.read_excel(filename)
	
	return cost

with header:
    st.title('Cost and Price Prediction of Edtech products by using machine learning techniques!')
    st.text('In this project I look into the Price_of_course.')

with dataset:
    st.header('cost_and_price dataset')
    st.text('I found this dataset on blablabla.com,...')

    cost = get_data("C:\Data science\8888\Project Dataset\cost_and_price.xlsx")

st.subheader('Pick-up location Price_of_course on the cost_and_price dataset')
Price_of_course = pd.DataFrame(cost['Price_of_course'].value_counts()).head()
st.bar_chart(Price_of_course)

with features:
    st.header('the features I created')
    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
    st.markdown('* **second feature:** I created this feature because of the this... I calculated it using this logic..')

with modelTraining:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')
    
from sklearn.ensemble import RandomForestRegressor    
sel_col, disp_col = st.beta_columns(2)
max_depth = sel_col.slider('what should be the max_depth of the model?', min_value = 10000, max_value = 30000, value = 3000, step = 14)
n_estimators = sel_col.selectbox('how many trees should there be', options=[1000,6000,10000,'No limit'], index= 0)

sel_col.text('Here is a list of features in data:')
sel_col.write(cost.beta_columns)

input_feature = sel_col.text_input('which feature should be used as input feature', 'Price_of_course')

if n_estimators = 'No limit':
	regr = RandomForestRegressor(max_depth=max_depth)
else:
	regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

x = cost[['Price_of_course']]
y = cost[['Study_material_price']]

regr.fit(x,y)
prediction = regr.predict(y)

disp_col.subheader = ['Mean absolute error of the model is:']
disp_col.write(mean_absolute_error(y, prediction))

disp_col.subheader('Mean squared error of the model is:')
disp_col.write(mean_squared_error(y, prediction))

disp_col.subheader('R squared score of the model is:')
disp_col.write(r2_score(y, prediction))

