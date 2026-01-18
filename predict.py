"""
FreshScanAI - Inference & Prediction Module
Real-time food freshness detection
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from config import MODEL_CONFIG, CLASS_CONFIG

class FreshScanAIPredictor:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or MODEL_CONFIG['model_path'])
        self.model = None
        self.class_names = MODEL_CONFIG['class_names']
        self.img_size = 224
        self.class_indices = None
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load class indices if available
            class_indices_path = self.model_path.parent / 'class_indices.json'
            if class_indices_path.exists():
                with open(class_indices_path, 'r') as f:
                    self.class_indices = json.load(f)
                # Reverse mapping
                self.class_names = [k for k, v in sorted(self.class_indices.items(), 
                                                         key=lambda x: x[1])]
                print(f"   Classes: {self.class_names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert('RGB')
            else:
                # Already a PIL Image
                image = image_path.convert('RGB')
            
            # Resize to model input size
            image_resized = image.resize((self.img_size, self.img_size), 
                                        Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array, image
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def predict(self, image_path):
        """Predict freshness from image"""
        # Preprocess image
        image_array, original_image = self.preprocess_image(image_path)
        
        if image_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        predicted_class = self.class_names[class_idx]
        
        # Get health advice
        class_config = CLASS_CONFIG[predicted_class]
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            },
            'emoji': class_config['emoji'],
            'icon': class_config['icon'],
            'status': class_config['status'],
            'recommendation': class_config['recommendation'],
            'risk_level': class_config['risk_level'],
            'health_risks': class_config['health_risks'],
            'prevention': class_config['prevention'],
            'color': class_config['color'],
            'hex_color': class_config['hex_color'],
            'storage_days': class_config['storage_days']
        }
        
        return result
    
    def visualize_prediction(self, image_path, result=None, save_path=None):
        """Visualize prediction with color-coded result"""
        if result is None:
            result = self.predict(image_path)
        
        if result is None:
            print("❌ Cannot visualize - prediction failed")
            return
        
        image = Image.open(image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Input Food Image', fontsize=14, fontweight='bold', pad=10)
        axes[0].axis('off')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['bottom'].set_visible(False)
        axes[0].spines['left'].set_visible(False)
        
        # Prediction result
        color_norm = tuple(c/255 for c in result['color'])
        
        # Title with emoji and class
        title_text = f"{result['emoji']} {result['predicted_class'].replace('_', ' ')}"
        
        axes[1].text(0.5, 0.88, title_text, 
                    ha='center', va='center', 
                    fontsize=26, fontweight='bold',
                    transform=axes[1].transAxes,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=result['hex_color'], 
                            edgecolor='none', alpha=0.9))
        
        # Status
        axes[1].text(0.5, 0.78, result['status'], 
                    ha='center', va='center', 
                    fontsize=13, fontweight='bold',
                    transform=axes[1].transAxes,
                    color='#333')
        
        # Confidence
        conf_text = f"Confidence: {result['confidence']*100:.1f}%"
        axes[1].text(0.5, 0.70, conf_text, 
                    ha='center', va='center', 
                    fontsize=13, transform=axes[1].transAxes,
                    color='#555')
        
        # Risk level
        risk_text = f"Risk Level: {result['risk_level']}"
        risk_color = '#4CAF50' if result['risk_level'] == 'None' else \
                    '#FFC107' if 'Medium' in result['risk_level'] else '#F44336'
        axes[1].text(0.5, 0.62, risk_text, 
                    ha='center', va='center', 
                    fontsize=12, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor=risk_color, 
                            alpha=0.3, edgecolor=risk_color, linewidth=2))
        
        # Recommendation
        axes[1].text(0.5, 0.48, "💡 Recommendation:", 
                    ha='center', va='center', 
                    fontsize=11, fontweight='bold',
                    transform=axes[1].transAxes,
                    color='#333')
        
        # Wrap text
        import textwrap
        wrapped_text = textwrap.fill(result['recommendation'], width=50)
        axes[1].text(0.5, 0.32, wrapped_text, 
                    ha='center', va='center', 
                    fontsize=9.5,
                    transform=axes[1].transAxes,
                    color='#555')
        
        # Storage info
        storage_text = f"📅 Storage: {result['storage_days']}"
        axes[1].text(0.5, 0.12, storage_text, 
                    ha='center', va='center', 
                    fontsize=10, style='italic',
                    transform=axes[1].transAxes,
                    color='#666')
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_facecolor('#F5F5F5')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_predict(self, image_dir):
        """Predict freshness for multiple images"""
        image_dir = Path(image_dir)
        results = []
        
        image_files = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg')) + \
                     list(image_dir.glob('*.png'))
        
        print(f"\n🔍 Processing {len(image_files)} images...")
        
        for image_path in image_files:
            result = self.predict(str(image_path))
            if result:
                results.append(result)
                print(f"   ✓ {image_path.name}: {result['predicted_class']} "
                     f"({result['confidence']*100:.1f}%)")
        
        print(f"\n✅ Processed {len(results)} images successfully!")
        
        return results
    
    def plot_probability_distribution(self, result, save_path=None):
        """Plot probability distribution for prediction"""
        probs = result['all_probabilities']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(probs.keys())
        probabilities = [probs[c] * 100 for c in classes]
        colors = [CLASS_CONFIG[c]['hex_color'] for c in classes]
        
        bars = ax.barh(classes, probabilities, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 1, i, f'{prob:.1f}%', 
                   va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Probability plot saved to {save_path}")
        
        plt.show()

def main():
    print("\n" + "="*70)
    print("🔥 FreshScanAI - INFERENCE MODULE".center(70))
    print("="*70)
    
    # Check if model exists
    model_path = Path(MODEL_CONFIG['model_path'])
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train the model first by running: python train_model.py")
        return
    
    # Initialize predictor
    predictor = FreshScanAIPredictor()
    
    # Example: Predict on a test image (you'll need to provide actual image path)
    test_dir = Path(DATA_CONFIG['processed_data_dir']) / 'test'
    
    # Find first available test image
    test_image = None
    for label in ['Fresh', 'Slightly_Spoiled', 'Rotten']:
        label_dir = test_dir / label
        if label_dir.exists():
            images = list(label_dir.glob('*.[jJ][pP][gG]')) + \
                    list(label_dir.glob('*.[pP][nN][gG]'))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        print(f"\n🔍 Predicting on: {test_image.name}")
        result = predictor.predict(test_image)
        
        if result:
            print("\n" + "="*70)
            print("📊 PREDICTION RESULT".center(70))
            print("="*70)
            print(f"\nClass:        {result['predicted_class']}")
            print(f"Status:       {result['status']}")
            print(f"Confidence:   {result['confidence']*100:.2f}%")
            print(f"Risk Level:   {result['risk_level']}")
            print(f"Storage:      {result['storage_days']}")
            print(f"\nRecommendation:\n  {result['recommendation']}")
            print(f"\nAll Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name:20s}: {prob*100:5.2f}%")
            
            # Visualize
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            predictor.visualize_prediction(
                test_image, result, 
                save_path=results_dir / 'prediction_example.png'
            )
            predictor.plot_probability_distribution(
                result,
                save_path=results_dir / 'probability_distribution.png'
            )
    else:
        print("\n⚠️ No test images found. Please run preprocessing.py first.")

if __name__ == "__main__":
    main()
