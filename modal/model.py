import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the base directory where your images are stored
base_dir = 'C:\\Users\\mraga\\Downloads\\Final_plant_detection\\Final_dataset\\'
num_classes = len(os.listdir(base_dir))  # Count the number of classes

# Load and prepare dataset with ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # or 'binary' based on your classes
)

# Create and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use num_classes here
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)
model.save('model/plant_model.h5')
