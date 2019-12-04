import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,[3,3], activation  ='relu', padding = 'same'),
  tf.keras.layers.Conv2D(32,[3,3], activation  ='relu', padding = 'same', strides = [2,2]),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64,[1,1], activation  ='relu', padding = 'same'),
  tf.keras.layers.Conv2D(128,[1,1], activation  ='relu', padding = 'same', strides = [2,2]),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90)
#iter=datagen.flow(x_train, y_train)
#history = model.fit_generator(iter, steps_per_epoch=2000, epochs=10)

tf.keras.callbacks.EarlyStopping(monitor='val_loss')
history = model.fit(x_train, y_train, batch_size = 32, validation_split=.2, epochs=10)

# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")


import matplotlib.pyplot as plt


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

results = model.evaluate(x_test, y_test)

with open("experiment_data.txt",'a') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(str(history.history['acc'][-1]))
    f.write(str(results[-1]))
