from __future__ import print_function

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Flatten

np.random.seed(123)
X = np.load('../generated_games/features-200.npy')
Y = np.load('../generated_games/labels-200.npy')

samples = X.shape[0]
size = 9
input_shape = (size, size, 1)

X = X.reshape(samples, size, size, 1)

train_samples = 10000
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='sigmoid',
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='sigmoid'))

model.add(Flatten())

model.add(Dense(128, activation='sigmoid'))
model.add(Dense(size * size, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
