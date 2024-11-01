import tensorflow as tf
import numpy as np

input_data = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
output_data = np.array([0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=3, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05),
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(input_data, output_data, epochs=100)

loss, accuracy = model.evaluate(input_data, output_data, verbose=0)
print("Loss: ", loss*100, "%")
print("Accuracy: ", accuracy*100, "%")

predictions = model.predict(input_data)
print(predictions)

