"""
FreshScanAI - Model Evaluation Module
Comprehensive metrics and visualizations
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from config import DATA_CONFIG, MODEL_CONFIG, CLASS_CONFIG

class ModelEvaluator:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or MODEL_CONFIG['model_path'])
        self.model = None
        self.class_names = MODEL_CONFIG['class_names']
        self.img_size = DATA_CONFIG['img_size']
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✅ Model loaded for evaluation")
    
    def load_test_data(self):
        """Load test dataset"""
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        test_data = test_datagen.flow_from_directory(
            Path(DATA_CONFIG['processed_data_dir']) / 'test',
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Test data loaded: {test_data.samples} samples")
        
        return test_data
    
    def evaluate(self, test_data):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("📊 EVALUATING MODEL".center(70))
        print("="*70)
        
        results = self.model.evaluate(test_data, verbose=0)
        
        # Results from evaluate are [loss, accuracy] - model wasn't compiled with metrics
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1])
        }
        
        # Calculate precision/recall/f1 from confusion matrix later
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
        
        print(f"\n📈 Test Metrics:")
        print(f"   Loss:      {metrics['loss']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']*100:.2f}%")
        print(f"   Recall:    {metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        
        return metrics
    
    def get_predictions(self, test_data):
        """Get predictions on test set"""
        print("\n🔄 Generating predictions...")
        
        y_true = test_data.classes
        y_pred_proba = self.model.predict(test_data, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("✅ Predictions generated")
        
        return y_true, y_pred, y_pred_proba
    
    def compute_detailed_metrics(self, y_true, y_pred):
        """Compute per-class metrics"""
        metrics_dict = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'f1_score_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        }
        
        return metrics_dict
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[c.replace('_', '\n') for c in self.class_names],
                   yticklabels=[c.replace('_', '\n') for c in self.class_names],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14, 'fontweight': 'bold'},
                   linewidths=2, linecolor='white')
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        
        # Add accuracy for each class
        for i in range(len(self.class_names)):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum() * 100
                plt.text(i, i - 0.3, f'{acc:.1f}%', 
                        ha='center', va='center', 
                        fontsize=10, color='darkgreen', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def print_classification_report(self, y_true, y_pred, save_path=None):
        """Print and save classification report"""
        print("\n" + "="*70)
        print("📋 CLASSIFICATION REPORT".center(70))
        print("="*70)
        
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      digits=4)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("FreshScanAI - Classification Report\n")
                f.write("="*70 + "\n\n")
                f.write(report)
            print(f"✅ Classification report saved to {save_path}")
        
        return report
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path=None):
        """Plot per-class performance metrics"""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', 
                      color='#4CAF50', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', 
                      color='#2196F3', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                      color='#FF9800', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        ax.set_xlabel('Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in self.class_names])
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Per-class metrics saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curves for each class"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, y_true, y_pred, y_pred_proba, metrics):
        """Generate comprehensive evaluation report"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        report = {
            'overall_metrics': metrics,
            'per_class_metrics': {
                'precision': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
                'recall': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
                'f1_score': f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'class_names': self.class_names
        }
        
        # Save report
        report_path = results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"✅ Evaluation report saved to {report_path}")
        
        return report

def main():
    print("\n" + "="*70)
    print("🔥 FreshScanAI - MODEL EVALUATION".center(70))
    print("="*70)
    
    # Check if model exists
    model_path = Path(MODEL_CONFIG['model_path'])
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train the model first by running: python train_model.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate on test set
    metrics = evaluator.evaluate(test_data)
    
    # Get predictions
    y_true, y_pred, y_pred_proba = evaluator.get_predictions(test_data)
    
    # Compute detailed metrics
    detailed_metrics = evaluator.compute_detailed_metrics(y_true, y_pred)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Print classification report
    evaluator.print_classification_report(
        y_true, y_pred, 
        save_path=results_dir / 'classification_report.txt'
    )
    
    # Plot visualizations
    print("\n📊 Generating visualizations...")
    evaluator.plot_confusion_matrix(
        y_true, y_pred, 
        save_path=results_dir / 'confusion_matrix.png'
    )
    evaluator.plot_per_class_metrics(
        y_true, y_pred, 
        save_path=results_dir / 'per_class_metrics.png'
    )
    evaluator.plot_roc_curves(
        y_true, y_pred_proba,
        save_path=results_dir / 'roc_curves.png'
    )
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report(y_true, y_pred, y_pred_proba, detailed_metrics)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()
