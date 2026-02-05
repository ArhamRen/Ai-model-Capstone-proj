# Overview: Trains one general model using your pre-split dataset (train/val/test). No env-specific bias/oversampling – add back later.

import os
import shutil
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

USER_DATASET_DIR = r'C:\Users\arham\Downloads\AI model data\dataset'

CATEGORIES = ['organic', 'plastic', 'paper']

OVERSAMPLE_FACTOR = 1 

def prepare_dataset():
    processed_dir = './processed_general/'
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    

    for split in ['train', 'val']:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(processed_dir, split, cat), exist_ok=True)
    
    for split in ['train', 'val']:
        for orig_class in CATEGORIES:
            src_dir = os.path.join(USER_DATASET_DIR, split, orig_class)
            if not os.path.exists(src_dir):
                print(f"Warning: Folder {src_dir} not found – skipping {orig_class} for {split}.")
                continue
            
            images = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(images) == 0:
                print(f"No images in {src_dir} – add some!")
                continue
            
       
            dest_split_dir = train_dir if split == 'train' else val_dir
            for i, img in enumerate(images):
                base_name = os.path.basename(img)
                for dup in range(OVERSAMPLE_FACTOR):
                    dest = os.path.join(dest_split_dir, orig_class, f"{i}_dup{dup}_{base_name}")
                    shutil.copy(img, dest)
    
    print("General dataset prepared.")


if not os.path.exists('./processed_general/'):
    prepare_dataset()


def count_images():
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()} split:")
        for cat in CATEGORIES:
            dir_path = os.path.join(USER_DATASET_DIR, split, cat)
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {cat}: {count} images")
            else:
                print(f"  {cat}: Missing folder")
        
        count_images()


IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS = 10 
FINE_TUNE_EPOCHS = 5

def train_model():
    processed_dir = './processed_general/'
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30, 
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    NUM_CLASSES = len(CATEGORIES)
    
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  
    x = Dense(512, activation='relu')(x) 
    predictions = Dense(NUM_CLASSES, activation='softmax')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    

    model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        epochs=EPOCHS
    )
    

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False  
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        epochs=FINE_TUNE_EPOCHS
    )
    
    # Save
    model.save('waste_classifier.h5')
    print("Model saved as waste_classifier.h5")
    
    # Convert to TFLite for Pi
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('waste_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite saved as waste_classifier.tflite")
    
    return model  # Return for evaluation


print("\nTraining general model...")
model = train_model()

# Step 4: Evaluate on Test Set (final check on unseen data)
def evaluate_on_test(model):
    test_dir = os.path.join(USER_DATASET_DIR, 'test')  # Your original test – no oversample needed
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Keep order for accurate eval
    )
    
    # Predict and score
    loss, acc = model.evaluate(test_generator)
    print(f"General Model - Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

print("\nEvaluating general model on test set...")
evaluate_on_test(model)

