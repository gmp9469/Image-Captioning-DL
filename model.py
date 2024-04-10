import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sys

sys.setrecursionlimit(10000)  

try:
    ds_train, ds_test, info = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True, with_info=True)
    def preprocess_image(image, label):
        image = tf.image.resize(image, (300, 300))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    ds_train = ds_train.map(preprocess_image).shuffle(1000).batch(32)
    ds_test = ds_test.map(preprocess_image).batch(32)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds_train, epochs=10, validation_data=ds_test)
    model.save('RPSmodel.h5')

except Exception as e:
    print("An error occurred during training:", e)
