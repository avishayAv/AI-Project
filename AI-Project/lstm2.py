import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout


"""combiner = 'combiner.csv'
combiner_df = pd.read_csv(combiner)

X = combiner_df.loc[:, combiner_df.columns != 'Close_1'].drop(columns=['Date','label'])
y = combiner_df['Close_1']

temp = combiner_df.to_numpy()"""

combiner = 'SP500.csv'
combiner_df = pd.read_csv(combiner)

X = combiner_df.loc[:, combiner_df.columns != 'Close'].drop(columns=['Date','Adj Close'])
y = combiner_df['Close']

X = X.to_numpy()
y = y.to_numpy()

#X_train, a ,b ,c,y_train= temp.T
#X_train=np.vstack([a,b]).T

X_train = np.reshape(X, (X.shape[0], 1,X.shape[1]))
X_train,X_test = X_train[:6000], X_train[6000:]
#y_train = np.reshape(y, (466, 5, 1))
y_train , y_test = y[:6000], y[6000:]
regressor = Sequential()

regressor.add(LSTM(units=10, input_shape=(1,X_train.shape[1]),return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

"""regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))"""

regressor.add(Dense(units=10,input_shape=(1,X_train.shape[1])))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=25, batch_size=32)

predicted_stock_price = regressor.predict(X_test)
a=1