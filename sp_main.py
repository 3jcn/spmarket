import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import math
import pandas_datareader as web
from datetime import date
from datetime import datetime
from PIL import Image
plt.style.use('fivethirtyeight')

# author: Thomas Nguyen; modified from Dataprofessor & Computer Science; March 6, 2021

st.markdown("""
	<style>
	.main{
	background-color: #F0F7F8;
	}
        .stButton>button {
        background-color: #0000ff;
        color: #ffffff;
        }
        .stTextInput>div>div>input {
        background-color: yellow;
        color: brown;
        }
	</style>
	""",
	unsafe_allow_html=True
)

@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  fig = plt.figure()
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='blue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price (USD)', fontweight='bold')
  return st.pyplot(fig)

df = load_data()
sector = df.groupby('GICS Sector')

image = Image.open('banner.png')
st.image(image,use_column_width=True)

#html_temp = """
#	<div style="background-color:brown; padding:1px">
#	<h1 style="color:white; text-align:center;">S&P 500 App</h1>
#	</div>
#	"""
#st.markdown(html_temp,unsafe_allow_html=True)
st.write('@author: [Thomas Nguyen](https://www.new3jcn.com)')
#st.title('S&P 500 App')

st.markdown("""
    This app shows **the stock closing price** of a company in **S&P 500** year-to-date 2021 and from 1/1/2012 to today!
    This app also predicts the stock closing price for the company for the next stock market open day.
    * **Data source for 2021:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
    * **Data source for 2012-2020:** [Yahoo](https://finance.yahoo.com/quote/ATVI/history?p=ATVI) 
    """)

# show data:
# df


# Sidebar - Sector selection:
st.sidebar.header('Control Panel:')
# Find all distinct sectors:
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
all_sectors = st.sidebar.multiselect('All sectors in S&P 500:', sorted_sector_unique, sorted_sector_unique)

# Get name of selected sector:
selected_sector = st.sidebar.selectbox('Choose a sector:', options = sorted_sector_unique)

## df_selected_sector = df[ (df['GICS Sector'].isin(all_sectors)) ]
## company_in_selected_sector = st.sidebar.selectbox('Choose a company:', options = df_selected_sector)

df_selected_sector = df[ (df['GICS Sector']==selected_sector) ]
st.subheader('List of all companies in your selected sector:  ' + selected_sector)
df_selected_sector

# create a list of all companies in the selected sector:
sorted_companies = sorted(df_selected_sector['Symbol'].unique() )

final_selected = df[ (df['GICS Sector'].isin(sorted_companies)) ]
company_name = st.sidebar.selectbox('Choose a company in the sector:', options = sorted_companies)

data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# Plot the selected company's closing price:
# Get the selected company's name:
comp_name = df[df['Symbol']==company_name]
name = comp_name['Security'].to_string()
st.subheader("Display the stock closing price of ' " + name[3:] + "' year-to-date 2021:")

index = df.index[df['Symbol']==company_name]
price_plot(df_selected_sector.Symbol[index[0]])

#if st.button('Show Graph:'):
    #price_plot(df_selected_sector.Symbol[index[0]])

##############  ML #############

# get current date:
today = date.today()
now = datetime.now()
data = web.DataReader(company_name,data_source='yahoo',start='2012-01-01', end=today)
st.subheader("Display the stock closing price of '" + name[3:] + "' from 01/01/2012 to " + now.strftime("%m/%d/%Y"))

dt = pd.DataFrame(data['Close'])
dt['Date'] = dt.index
fig = plt.figure()
plt.fill_between(dt.Date, dt.Close, color='skyblue', alpha=0.3)
plt.plot(dt.Date, dt.Close, color='blue', alpha=0.8)
plt.xticks(rotation=90)
plt.title(company_name, fontweight='bold')
#plt.xlabel('Time', fontweight='bold')
plt.ylabel('Closing Price (USD)', fontweight='bold')
st.pyplot(fig)

st.subheader("Machine Learning Model Prediction:")
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
#from keras.layers import Dense, LSTM
#from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

data = data.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len,:] 
#st.write(train_data) = 1812
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)
# create three dimensions data: x_train.shape[0]=rows x_train.shape[1]=columns
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1, epochs=1)

#create testing data:
test_data = scaled_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
# create 3 dim for the model LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2 )
st.subheader("Root Mean Squared Error of this model:")
st.write(rmse)
score=r2_score(y_test,predictions)
st.subheader("R2 Score of this model:")
st.write(score)


#Get the quote:
apple_quote = web.DataReader(company_name, data_source='yahoo',start='2012-01-01',end=today)
#create a new dataframe
new_df=apple_quote.filter(['Close'])
#get last 60 days closing price and convert data to array
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
pred_price = model.predict(X_test)
# undo scaling
pred_price = scaler.inverse_transform(pred_price)

st.subheader("Predicting the stock closing price of the company for the next stock market open day:")
st.write(pred_price)
