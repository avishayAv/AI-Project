import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


sp500 = 'SP500.csv'
sp500_df = pd.read_csv(sp500)
sp500_df.drop(columns=['Open', 'Adj Close'], inplace=True)  # SP500 : Date | High | Low | Close | Volume
temp = sp500_df.to_numpy()

X_train, a ,b ,c,y_train= temp.T
X_train=np.vstack([a,b]).T

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train,X_test = X_train[:6000], X_train[6000:]
y_train , y_test = y_train[:6000], y_train[6000:]
regressor = Sequential()

regressor.add(LSTM(units=50, input_shape=(X_train.shape[1], 1),return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=10, batch_size=32)

predicted_stock_price = regressor.predict(X_test)
a=1