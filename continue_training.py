"""
Continue Training FreshScanAI Model
Load the pre-trained model and continue training on your own dataset
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json

# Configuration
DATA_DIR = Path('data/processed')
MODEL_PATH = 'models/freshscan_model.h5'
NEW_MODEL_PATH = 'models/freshscan_model_retrained.h5'
BATCH_SIZE = 32
EPOCHS = 10
INITIAL_LR = 0.0001  # Lower learning rate for fine-tuning

def load_existing_model():
    """Load the pre-trained FreshScanAI model"""
    print("📦 Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    model.summary()
    return model

def create_data_generators():
    """Create data generators for training"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_data = val_datagen.flow_from_directory(
        DATA_DIR / 'val',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_data, val_data

def continue_training(model, train_data, val_data):
    """Continue training the model"""
    print("\n🔥 Continuing training with fine-tuning...")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            NEW_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Training complete! Model saved to {NEW_MODEL_PATH}")
    return history

def main():
    """Main training pipeline"""
    print("="*70)
    print("   🔥 FreshScanAI - Continue Training")
    print("="*70)
    
    # Check if data exists
    if not (DATA_DIR / 'train').exists():
        print("❌ Training data not found!")
        print("Please ensure data is in:", DATA_DIR / 'train')
        print("\nTo prepare data, run: python preprocessing.py")
        return
    
    # Load existing model
    model = load_existing_model()
    
    # Load data
    print("\n📊 Loading training data...")
    train_data, val_data = create_data_generators()
    
    # Continue training
    history = continue_training(model, train_data, val_data)
    
    print("\n" + "="*70)
    print("✅ Training completed successfully!")
    print("="*70)
    print(f"\n📁 New model saved at: {NEW_MODEL_PATH}")
    print("\nTo use the retrained model in the app:")
    print("1. Rename the old model or backup it")
    print("2. Rename freshscan_model_retrained.h5 to freshscan_model.h5")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
