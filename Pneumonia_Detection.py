import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import cv2
import math

print("=== PNEUMONIA DETECTION MODEL ===")

# ===== 90%+ ACCURACY MODEL =====
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

# Use DenseNet121 - proven best for medical images
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)
base_model.trainable = False  # Keep pre-trained weights frozen

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Low learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===== DATA PATHS =====
train_dir = r'C:\Users\Ammar N. Ahmed\OneDrive - Thecatgroup\Desktop\Pneumonia_Detection-ougvhy\dataset\train'
test_dir = r'C:\Users\Ammar N. Ahmed\OneDrive - Thecatgroup\Desktop\Pneumonia_Detection-ougvhy\dataset\test'
val_dir = r'C:\Users\Ammar N. Ahmed\OneDrive - Thecatgroup\Desktop\Pneumonia_Detection-ougvhy\dataset\val'

print("=== DATASET PATHS ===")
print("Train directory:", train_dir)
print("Test directory:", test_dir)
print("Val directory:", val_dir)

# Verify paths exist
print("\n=== PATH VERIFICATION ===")
print("Train exists:", os.path.exists(train_dir))
print("Test exists:", os.path.exists(test_dir))
print("Val exists:", os.path.exists(val_dir))

# ===== DATA GENERATORS =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("\n=== CREATING DATA GENERATORS ===")

# ===== CREATE GENERATORS FIRST =====
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    train_dir,  # Use same directory as training
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Extra validation from val folder
extra_val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ===== CALCULATE CLASS WEIGHTS =====
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights to handle imbalance 
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# ===== CALCULATE STEPS =====
train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
val_steps = math.ceil(val_generator.samples / val_generator.batch_size)
test_steps = math.ceil(test_generator.samples / test_generator.batch_size)

print(f"\n=== TRAINING STEPS ===")
print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps: {val_steps}")
print(f"Test steps: {test_steps}")

# ===== CALLBACKS =====
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Changed from 'val_loss'
    patience=15,
    restore_best_weights=True,
    mode='max',  # Added this
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Changed from 'val_loss'
    factor=0.5,
    patience=8,
    min_lr=0.00001,
    mode='max',  # Added this
    verbose=1
)

# ===== MODEL TRAINING =====
print("\n=== STARTING TRAINING ===")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# ===== FINE-TUNING FOR 90%+ =====

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile with very low learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Very low LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("=== FINE-TUNING FOR 90%+ ACCURACY ===")

# Fine-tune for a few more epochs
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=10,  # Just 10 more epochs
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Final evaluation
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
print(f'FINAL TEST ACCURACY: {test_accuracy*100:.2f}%')

# ===== MODEL EVALUATION =====
print("\n=== MODEL EVALUATION ===")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

# Evaluate on extra validation set
if extra_val_generator.samples > 0:
    val_loss, val_accuracy = model.evaluate(extra_val_generator, verbose=1)
    print(f'Extra Validation Accuracy: {val_accuracy*100:.2f}%')
    print(f'Extra Validation Loss: {val_loss:.4f}')

# ===== SAVE MODEL =====
print("\n=== SAVING MODEL ===")
# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
model.save('models/Pneumonia_Model.h5')
print("Model saved successfully!")

# ===== PLOT RESULTS =====
print("\n=== GENERATING PLOTS ===")
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("\n=== TRAINING COMPLETE ===")
print("Check 'training_history.png' for accuracy/loss plots")
print("Model saved as 'models/Pneumonia_Model.h5'") 