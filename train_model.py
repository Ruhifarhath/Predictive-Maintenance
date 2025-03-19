import tensorflow as tf

# Example Model (Modify as per your dataset)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming you have X_train and y_train ready
# model.fit(X_train, y_train, epochs=10, batch_size=32)

model.save("failure_detection_model.h5")
