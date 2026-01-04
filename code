#importing all required libirary
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#getting NIFTY-50 10 years data
end_date = datetime.today()-timedelta(days=1)
start_date = end_date - timedelta(days=365*10)
data = yf.download("^NSEI", start=start_date, end=end_date)
data.reset_index(inplace=True)
#checking if on (t+1) day stock price increased than t day{this would be used for LOGISTIC regression}
data['Target'] = (data['Close'] < data['Close'].shift(-1)).astype(bool)
#removing any NaN value in dataset
data.dropna(inplace=True)

#for ML training multiple features are introduced for better accuracy of the model
#return is one day return
data['Return'] = data['Close'].pct_change()
#for intraday
data['High_Low'] = data['High'] - data['Low']
#finding bullish or bearish behaviour
data['Open_Close'] = data['Open'] - data['Close']
#EWM is exponential moving accuracy used to boost accuracy
data['ewm5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['ewm10'] = data['Close'].ewm(span=10, adjust=False).mean()

#removing NaN and infinity values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

features = ['Return', 'High_Low', 'Open_Close', 'ewm5', 'ewm10']

X = data[features]
y = data['Target']
#using 60% data for training and rest for testing
split = int(0.6 * len(data))
#for training data before 60%
X_train=X.iloc[:split]
y_train=y.iloc[:split]
# for data from 60 to 100%
X_test=X.iloc[split:]
y_test=y.iloc[split:]

#making ML Pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#using last column for prediction
last_day=X.iloc[[-1]]
prediction=model.predict(last_day)

if prediction[0]==True:
    print("Yes,it would increase")
else:
    print("No,it would decrease")
#ploting graph
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('NIFTY-50 Close Price')
plt.grid()
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))

