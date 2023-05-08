import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for layer in model.layers[:len(base_model.layers)]:
    layer.trainable = False
for layer in model.layers[len(base_model.layers):]:
    layer.trainable = True
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
