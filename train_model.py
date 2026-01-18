"""
FreshScanAI - Model Training Module
Transfer Learning with MobileNetV2 + TensorFlow
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG

class FreshScanAIModel:
    def __init__(self, config, model_config, training_config):
        self.config = config
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.history = None
        self.class_indices = None
        
    def load_data(self):
        """Load and augment training data"""
        print("\n📥 Loading data...")
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rescale=AUGMENTATION_CONFIG['rescale'],
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            brightness_range=AUGMENTATION_CONFIG['brightness_range'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode'],
            shear_range=AUGMENTATION_CONFIG['shear_range']
        )
        
        # Validation/Test (no augmentation, only rescaling)
        val_datagen = ImageDataGenerator(rescale=AUGMENTATION_CONFIG['rescale'])
        test_datagen = ImageDataGenerator(rescale=AUGMENTATION_CONFIG['rescale'])
        
        # Load data from directories
        train_data = train_datagen.flow_from_directory(
            Path(self.config['processed_data_dir']) / 'train',
            target_size=(self.config['img_size'], self.config['img_size']),
            batch_size=self.training_config['batch_size'],
            class_mode='categorical',
            shuffle=True
        )
        
        val_data = val_datagen.flow_from_directory(
            Path(self.config['processed_data_dir']) / 'val',
            target_size=(self.config['img_size'], self.config['img_size']),
            batch_size=self.training_config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        test_data = test_datagen.flow_from_directory(
            Path(self.config['processed_data_dir']) / 'test',
            target_size=(self.config['img_size'], self.config['img_size']),
            batch_size=self.training_config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_indices = train_data.class_indices
        
        print("✅ Data loaded successfully!")
        print(f"   Classes: {list(self.class_indices.keys())}")
        print(f"   Train samples: {train_data.samples}")
        print(f"   Val samples: {val_data.samples}")
        print(f"   Test samples: {test_data.samples}")
        
        # Save class indices
        with open(Path(MODEL_CONFIG['model_path']).parent / 'class_indices.json', 'w') as f:
            json.dump(self.class_indices, f, indent=4)
        
        return train_data, val_data, test_data
    
    def build_model(self):
        """Build transfer learning model with MobileNetV2"""
        print("\n🏗️  Building model architecture...")
        
        # Load pre-trained MobileNetV2 (ImageNet weights)
        base_model = MobileNetV2(
            input_shape=(self.config['img_size'], self.config['img_size'], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights (transfer learning)
        base_model.trainable = False
        
        # Build model using Functional API (TF 2.12 compatible)
        inputs = tf.keras.Input(shape=(self.config['img_size'], self.config['img_size'], 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.Dense(256, activation='relu', name='dense1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        outputs = layers.Dense(self.model_config['num_classes'], activation='softmax', name='output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='FreshScanAI')
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.training_config['learning_rate']
            ),
            loss=self.training_config['loss_function'],
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        
        print("✅ Model built successfully!")
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def train(self, train_data, val_data):
        """Train the model"""
        print("\n" + "="*70)
        print("🚀 TRAINING MODEL".center(70))
        print("="*70)
        
        # Create logs directory
        log_dir = Path('logs') / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['reduce_lr_factor'],
                patience=self.training_config['reduce_lr_patience'],
                min_lr=self.training_config['min_lr'],
                verbose=1
            ),
            ModelCheckpoint(
                self.model_config['model_path'],
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Train model
        print(f"\n⏱️ Starting training...")
        print(f"   Epochs: {self.training_config['epochs']}")
        print(f"   Batch size: {self.training_config['batch_size']}")
        print(f"   Learning rate: {self.training_config['learning_rate']}")
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        # Save training history
        history_path = Path(MODEL_CONFIG['model_path']).parent / 'training_history.json'
        with open(history_path, 'w') as f:
            history_dict = {key: [float(val) for val in values] 
                          for key, values in self.history.history.items()}
            json.dump(history_dict, f, indent=4)
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("📊 EVALUATING MODEL".center(70))
        print("="*70)
        
        results = self.model.evaluate(test_data, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        metrics['f1_score'] = f1_score
        
        print(f"\n📈 Test Results:")
        print(f"   Loss:      {metrics['loss']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']*100:.2f}%")
        print(f"   Recall:    {metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        
        # Save metrics
        metrics_path = Path(MODEL_CONFIG['model_path']).parent / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def save_model(self):
        """Save trained model"""
        self.model.save(self.model_config['model_path'])
        print(f"\n💾 Model saved to {self.model_config['model_path']}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("⚠️ No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], 
                       label='Train', linewidth=2, marker='o')
        axes[0, 0].plot(self.history.history['val_accuracy'], 
                       label='Validation', linewidth=2, marker='s')
        axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], 
                       label='Train', linewidth=2, marker='o')
        axes[0, 1].plot(self.history.history['val_loss'], 
                       label='Validation', linewidth=2, marker='s')
        axes[0, 1].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], 
                       label='Train', linewidth=2, marker='o')
        axes[1, 0].plot(self.history.history['val_precision'], 
                       label='Validation', linewidth=2, marker='s')
        axes[1, 0].set_title('Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], 
                       label='Train', linewidth=2, marker='o')
        axes[1, 1].plot(self.history.history['val_recall'], 
                       label='Validation', linewidth=2, marker='s')
        axes[1, 1].set_title('Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
        
        plt.show()

def main():
    print("\n" + "="*70)
    print("🔥 FreshScanAI - MODEL TRAINING (10 EPOCHS - FAST)".center(70))
    print("="*70)
    
    # Initialize model trainer
    trainer = FreshScanAIModel(DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG)
    
    # Override epochs to 10 for faster training
    trainer.training_config['epochs'] = 10
    print(f"\n⚙️  Training Configuration: {trainer.training_config['epochs']} epochs (fast training)")
    
    # Load data
    train_data, val_data, test_data = trainer.load_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train(train_data, val_data)
    
    # Evaluate on test set
    trainer.evaluate(test_data)
    
    # Save model
    trainer.save_model()
    
    # Plot history
    trainer.plot_training_history()
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()
