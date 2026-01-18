"""
FreshScanAI Configuration File
All project settings and configurations
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Data Configuration
DATA_CONFIG = {
    'raw_data_dir': str(DATA_DIR / 'raw'),
    'processed_data_dir': str(DATA_DIR / 'processed'),
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'img_size': 224,
    'img_format': 'RGB',
    'seed': 42
}

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'freshscan_model',
    'framework': 'tensorflow',
    'base_model': 'MobileNetV2',
    'num_classes': 3,
    'class_names': ['Fresh', 'Slightly_Spoiled', 'Rotten'],
    'model_path': str(MODEL_DIR / 'freshscan_model.h5'),
    'weights': 'imagenet',
    'include_top': False
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    'validation_split': 0.2
}

# Class Configuration with Health Information
CLASS_CONFIG = {
    'Fresh': {
        'color': (0, 255, 0),          # Green (RGB)
        'hex_color': '#4CAF50',
        'emoji': '✅',
        'status': 'SAFE TO CONSUME',
        'recommendation': 'Safe to consume. Store in a cool, dry place. Refrigerate for extended freshness.',
        'storage_days': '0-2 days',
        'risk_level': 'None',
        'age_range': '0-2 days',
        'health_risks': 'No health risks',
        'prevention': 'Store properly to maintain freshness',
        'icon': '🟢'
    },
    'Slightly_Spoiled': {
        'color': (255, 193, 7),        # Amber (RGB)
        'hex_color': '#FFC107',
        'emoji': '⚠️',
        'status': 'USE WITH CAUTION',
        'recommendation': 'Use with caution. Cook thoroughly at >165°F (74°C). Risk of mild food poisoning. Best to discard if unsure.',
        'storage_days': '2-5 days',
        'risk_level': 'Low-Medium',
        'age_range': '2-5 days',
        'health_risks': 'Mild stomach upset, nausea, diarrhea',
        'prevention': 'Cook thoroughly or discard immediately',
        'icon': '🟡'
    },
    'Rotten': {
        'color': (244, 67, 54),        # Red (RGB)
        'hex_color': '#F44336',
        'emoji': '🛑',
        'status': 'DO NOT CONSUME',
        'recommendation': 'DO NOT CONSUME under any circumstances. High risk of severe food poisoning, bacterial infection (E.coli, Salmonella), and hospitalization.',
        'storage_days': '>5 days',
        'risk_level': 'CRITICAL',
        'age_range': '>5 days',
        'health_risks': 'Severe food poisoning, bacterial infections, vomiting, fever, hospitalization',
        'prevention': 'Discard immediately. Wash hands after handling.',
        'icon': '🔴'
    }
}

# Food Categories (14 types)
FOOD_CATEGORIES = [
    'apple', 'banana', 'bellpepper', 'carrot', 'cucumber', 
    'grape', 'guava', 'jujube', 'mango', 'orange', 
    'pomegranate', 'potato', 'strawberry', 'tomato'
]

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest',
    'rescale': 1./255,
    'shear_range': 0.1
}

# Evaluation Configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
    'save_confusion_matrix': True,
    'save_classification_report': True,
    'visualization_dpi': 300,
    'results_dir': str(RESULTS_DIR)
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    'streamlit_port': 8501,
    'max_upload_size': 10,  # MB
    'confidence_threshold': 0.6,
    'model_confidence_decimal_places': 2,
    'allowed_extensions': ['jpg', 'jpeg', 'png', 'bmp'],
    'temp_dir': 'temp'
}

# UI Theme Configuration
UI_THEME = {
    'primary_color': '#4CAF50',
    'secondary_color': '#2196F3',
    'accent_color': '#FF9800',
    'danger_color': '#F44336',
    'warning_color': '#FFC107',
    'success_color': '#4CAF50',
    'background_color': '#FAFAFA',
    'text_color': '#212121'
}

# Health & Biosafety Information
BIOSAFETY_INFO = {
    'food_poisoning_symptoms': [
        'Nausea and vomiting',
        'Diarrhea',
        'Abdominal cramps',
        'Fever',
        'Headache',
        'Dehydration'
    ],
    'common_bacteria': [
        'Salmonella',
        'E. coli',
        'Listeria',
        'Staphylococcus aureus',
        'Clostridium botulinum'
    ],
    'prevention_tips': [
        'Check expiration dates regularly',
        'Store food at proper temperatures',
        'Keep refrigerator below 40°F (4°C)',
        'Cook food to safe internal temperatures',
        'Wash hands before handling food',
        'Use separate cutting boards for raw and cooked food',
        'When in doubt, throw it out'
    ],
    'emergency_contacts': {
        'Poison Control': '1-800-222-1222',
        'Emergency': '911'
    }
}

# Dataset Information
DATASET_INFO = {
    'name': 'Fruit & Vegetable Quality Dataset',
    'source': 'Kaggle',
    'url': 'https://www.kaggle.com/datasets/zlatan599/fruitquality1',
    'size': '5.21 GB',
    'total_images': '10000+',
    'categories': 14,
    'classes': 3,
    'format': 'JPG/PNG',
    'resolution': 'Variable'
}

# Project Information
PROJECT_INFO = {
    'name': 'FreshScanAI',
    'version': '1.0.0',
    'description': 'AI-Based Detection and Classification of Food Spoilage for Biosafety and Public Health Protection',
    'authors': ['Your Team Name'],
    'institution': 'Your College/University',
    'year': '2024-2025',
    'license': 'MIT'
}

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
