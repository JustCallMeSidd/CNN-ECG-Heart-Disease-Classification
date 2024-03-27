import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'D:\\Cardio cnn\\data\\train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',  
    shuffle=True 
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'D:\\Cardio cnn\\data\\test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'  
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(units=128, kernel_initializer='glorot_uniform', activation='relu'),
    Dense(units=2, kernel_initializer='glorot_uniform', activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=9,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)


loss, accuracy = model.evaluate(test_generator)


from keras.models import load_model
from keras.preprocessing import image
import numpy as np

img = image.load_img("D:\\Cardio cnn\\data\\use to Prediction\\Normal.png", target_size=(64, 64))  #"D:\\Cardio cnn\\data\\use to Prediction\\Normal.png"  "D:\\Cardio cnn\\data\\use to Prediction\\lb.png"


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


pred = model.predict(x)

y_pred = (pred > 0.5).astype(int)


index = ['Normal', 'left Bundle Branch block'] 
result = index[y_pred[0][0]]
print("Predicted Class:", result)