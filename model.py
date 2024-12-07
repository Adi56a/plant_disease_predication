import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Set path to the data directory
DATA_DIR = 'data'

# Directory paths
TRAIN_DIR = os.path.join(DATA_DIR, 'Train', 'Train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'Validation', 'Validation')
TEST_DIR = os.path.join(DATA_DIR, 'Test', 'Test')

# Set parameters
IMG_SIZE = 224  # Resize images to 224x224
BATCH_SIZE = 16  # Reduce batch size to alleviate memory issues
EPOCHS = 10
NUM_CLASSES = 3  # Healthy, Powdery, Rust

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize the images
    rotation_range=20,  # Data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Ensure shuffle=False for validation
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,  # We want to make individual predictions
    class_mode='categorical',
    shuffle=False  # Don't shuffle for testing
)

# Print the number of samples
print("Train Samples: ", train_generator.samples)
print("Validation Samples: ", validation_generator.samples)

# Build the CNN model
def create_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Explicit input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer with 3 classes
    ])
    return model

# Compile the model
model = create_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model checkpoint to save the best model
checkpoint = ModelCheckpoint('plant_disease_model.keras',  # Save the best model
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min', 
                             verbose=1)

# Adjust steps per epoch and validation steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save the final model
model.save('final_plant_disease_model.h5')

# Plot training history (accuracy/loss)
def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_history(history)
