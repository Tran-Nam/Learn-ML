import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K 

batch_size = 128
num_class = 10
epochs = 20

img_row, img_col = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_row, img_col, 1)
X_test = X_test.reshape(X_test.shape[0], img_row, img_col, 1)
input_shape = (img_row, img_col, 1)
# print(X_train[:10], y_train[:10])
X_train = X_train / 255. 
X_test = X_test / 255. 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_class, activation='softmax'))

print(model.summary())
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss', score[0])
print('Test acc', score[1])