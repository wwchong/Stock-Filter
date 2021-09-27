import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import pandas_datareader as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
import ta
from sklearn.linear_model import LinearRegression
from urllib.error import HTTPError

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#import your stock tickers list
ticker = pd.read_csv('Yourstocktickerlist.csv') 

#get data from yahoo finance and assign each stock a score based on different criteria
end = datetime.today() 
start = end + timedelta(-50)
industry = {}
potential = pd.DataFrame(data={"Ticker":0,"Score":0,"Industry":0},index=[0])
mark = 0
for i in ticker:
    score = 0
    try:
        tick = yf.Ticker(i)
        price = tick.history(start=start,end=end)
        price['Change'] = price['Close'].pct_change()
        price['Up_Down_Volume'] = np.zeros(len(price))
        ind = tick.info['industry']
        price['index'] = np.zeros(len(price))
        for k in range(1,len(price)):
            price['index'][i] = i

        #calculated the buying pressure and selling pressure of the stock based on its volume when its price rises and its volume when its price drops
        for p in range(1,len(price)):
            if price['Change'].iloc[p]>0 and price['Close'].iloc[p]>price['Open'].iloc[p]:
                price['Up_Down_Volume'].iloc[p] = price['Up_Down_Volume'].iloc[p-1] + price['Volume'].iloc[p]
            else:
                price['Up_Down_Volume'].iloc[p] = price['Up_Down_Volume'].iloc[p-1] - price['Volume'].iloc[p]
        
        if len(price)>0:
            #calculated the highest close price of the stock and its moving average of stock price and volume
            high_20 = max(price['Close'].iloc[-20:-1])
            vol_20 = price['Volume'].ewm(span=20).mean().dropna()
            ma_10 = price['Close'].rolling(window=10).mean().dropna()
            ema_20 = price['Close'].ewm(span=20).mean().dropna()
            ema_30 = price['Close'].ewm(span=30).mean().dropna()
            
            #if fulfill these requirements, the score of the industry of that stock will increase 1 point 
            if ma_10[-1]>ema_20[-1]>ema_30[-1] and price['Close'][-1]>high_20 and price['Up_Down_Volume'].iloc[-1]>0:
                try:
                    industry[ind]+=1
                except KeyError:
                    industry[ind]=1
            
            price.dropna(inplace=True)
            model = LinearRegression()
            high_trend = model.fit(np.array(price['index'][-5:]).reshape(-1,1),price['High'][-5:])
            model2 = LinearRegression()
            low_trend = model.fit(np.array(price['index'][-5:]).reshape(-1,1),price['Low'][-5:])

            #the score of the individual stock will increase 1 potin for each requirement fulfilled
            if ma_10[-1]>ema_20[-1]>ema_30[-1]:
                score += 1
            if price['Close'][-1]>high_20:
                score += 1
            if price['Volume'][-1]>vol_20[-1]:
                score += 1
            if price['Up_Down_Volume'][-1]>0:
                score += 1
            if high_trend.coef_>0 and low_trend.coef_>0:
                score += 1
        
        potential = pd.concat([potential,pd.DataFrame(data={"Ticker":i,"Score":score,"Industry":ind},index=[mark])])
        mark += 1

    except (RemoteDataError,HTTPError):
        print("RemoteDataError/HTTPError:"+ i)
    except KeyError:
        print("KeyError:"+ i)
    except IndexError:
        print("IndexError:" + i)
    except:
        print("Error:" + i)

#turn industry data into dataframe and sort it by the score of each industry
#higher score indicates stronger momentum of the stocks in that industry
industry = pd.DataFrame({"Score":industry.values()},industry.keys())
print(industry.sort_values('Score',ascending=False))

#print the first 30 individual stocks that have the highest score
print(potential.sort_values(by="Score",ascending=False).head(30))
