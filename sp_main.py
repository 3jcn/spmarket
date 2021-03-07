import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
# author: Thomas Nguyen; modified from github-Dataprofessor; March 6, 2021

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
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price (USD)', fontweight='bold')
  return st.pyplot(fig)

df = load_data()
sector = df.groupby('GICS Sector')

html_temp = """
	<div style="background-color:brown; padding:1px">
	<h1 style="color:white; text-align:center;">S&P 500 App</h1>
	</div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write('@author: [Thomas Nguyen](https://www.new3jcn.com)')
#st.title('S&P 500 App')

st.markdown("""
    This app shows **the stock closing price** of any company in **S&P 500** (year-to-date)!
    * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
    * **Content of data:** 
    """)

# show data:
df


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
st.subheader('List of all companies in your selected sector:')
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
st.subheader("Display the stock closing price of '" + name[3:] + "':")
index = df.index[df['Symbol']==company_name]
price_plot(df_selected_sector.Symbol[index[0]])

#if st.button('Show Graph:'):
    #price_plot(df_selected_sector.Symbol[index[0]])
