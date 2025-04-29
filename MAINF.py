import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Read the dataset from disk
df_train = pd.read_csv("data\mnist_train.csv")
df_test = pd.read_csv("data\mnist_test.csv")

# Split the training set into input and output data
x_train = df_train.iloc[:, 1:].values.astype('float32')
y_train = to_categorical(df_train.iloc[:, 0].values, num_classes=10)

# Split the test set into input and output data
x_test = df_test.iloc[:, 1:].values.astype('float32')
y_test = to_categorical(df_test.iloc[:, 0].values, num_classes=10)

# Reshape the data to the required shape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

batch_size = 400
n_epochs = 10

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epochs, batch_size=batch_size)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

# Make predictions
predictions = model.predict(x_test)
preds = np.argmax(predictions, axis=1)

# Calculate accuracy
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, preds)
print("Accuracy:", acc)

# Save the model
model.save("mnist_model.h5")
