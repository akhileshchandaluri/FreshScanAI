"""
FreshScanAI - Data Preprocessing Module
Downloads, organizes, and splits food spoilage dataset
"""

import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
from tqdm import tqdm
from config import DATA_CONFIG, FOOD_CATEGORIES

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config['raw_data_dir'])
        self.output_dir = Path(config['processed_data_dir'])
        self.img_size = config['img_size']
        self.train_split = config['train_split']
        self.val_split = config['val_split']
        self.test_split = config['test_split']
        self.seed = config['seed']
        
    def create_directories(self):
        """Create output directory structure"""
        print("\n📁 Creating directory structure...")
        for split in ['train', 'val', 'test']:
            for label in ['Fresh', 'Slightly_Spoiled', 'Rotten']:
                path = self.output_dir / split / label
                path.mkdir(parents=True, exist_ok=True)
        print("✅ Directory structure created!")
        
    def validate_image(self, img_path):
        """Validate if image file is readable"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            # Reopen to check it's actually a valid image
            with Image.open(img_path) as img:
                img.convert('RGB')
            return True
        except Exception as e:
            return False
            
    def preprocess_image(self, img_path, target_size=224):
        """Resize and normalize image"""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return None
            
    def organize_data_by_food(self):
        """
        Organize data: each food category should have fresh/ and rotten/ folders
        Creates Slightly_Spoiled class from 50% of rotten images
        """
        # Check for Unified_Dataset subdirectory
        unified_dataset = self.raw_dir / 'Unified_Dataset'
        if unified_dataset.exists():
            dataset_dir = unified_dataset
        else:
            dataset_dir = self.raw_dir
            
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            print(f"❌ Dataset directory not found or empty: {dataset_dir}")
            self._print_download_instructions()
            return False
            
        print("\n📂 Organizing dataset...")
        food_categories = [d for d in dataset_dir.iterdir() if d.is_dir()]
        total_images_processed = 0
        
        for food_dir in tqdm(food_categories, desc="Processing food categories"):
            food_name = food_dir.name
            
            fresh_dir = food_dir / 'fresh'
            rotten_dir = food_dir / 'rotten'
            
            # Process Fresh images
            if fresh_dir.exists():
                fresh_images = list(fresh_dir.glob('*.[jJ][pP][gG]')) + \
                              list(fresh_dir.glob('*.[jJ][pP][eE][gG]')) + \
                              list(fresh_dir.glob('*.[pP][nN][gG]'))
                
                # Validate images
                fresh_images = [img for img in fresh_images if self.validate_image(img)]
                
                if fresh_images:
                    # Split data
                    train_fresh, temp = train_test_split(
                        fresh_images, test_size=(1 - self.train_split), 
                        random_state=self.seed
                    )
                    val_size = self.val_split / (self.val_split + self.test_split)
                    val_fresh, test_fresh = train_test_split(
                        temp, test_size=(1 - val_size), random_state=self.seed
                    )
                    
                    # Copy images
                    self._copy_images(train_fresh, 'train', 'Fresh', food_name)
                    self._copy_images(val_fresh, 'val', 'Fresh', food_name)
                    self._copy_images(test_fresh, 'test', 'Fresh', food_name)
                    
                    total_images_processed += len(fresh_images)
            
            # Process Rotten images (split into Slightly_Spoiled & Rotten)
            if rotten_dir.exists():
                rotten_images = list(rotten_dir.glob('*.[jJ][pP][gG]')) + \
                               list(rotten_dir.glob('*.[jJ][pP][eE][gG]')) + \
                               list(rotten_dir.glob('*.[pP][nN][gG]'))
                
                # Validate images
                rotten_images = [img for img in rotten_images if self.validate_image(img)]
                
                if rotten_images:
                    # Split rotten into Slightly_Spoiled (50%) and Rotten (50%)
                    slightly_spoiled, rotten_rest = train_test_split(
                        rotten_images, test_size=0.5, random_state=self.seed
                    )
                    
                    # Split slightly_spoiled
                    train_ss, temp_ss = train_test_split(
                        slightly_spoiled, test_size=(1 - self.train_split), 
                        random_state=self.seed
                    )
                    val_size = self.val_split / (self.val_split + self.test_split)
                    val_ss, test_ss = train_test_split(
                        temp_ss, test_size=(1 - val_size), random_state=self.seed
                    )
                    
                    # Split rotten_rest
                    train_rot, temp_rot = train_test_split(
                        rotten_rest, test_size=(1 - self.train_split), 
                        random_state=self.seed
                    )
                    val_rot, test_rot = train_test_split(
                        temp_rot, test_size=(1 - val_size), random_state=self.seed
                    )
                    
                    # Copy Slightly_Spoiled images
                    self._copy_images(train_ss, 'train', 'Slightly_Spoiled', food_name)
                    self._copy_images(val_ss, 'val', 'Slightly_Spoiled', food_name)
                    self._copy_images(test_ss, 'test', 'Slightly_Spoiled', food_name)
                    
                    # Copy Rotten images
                    self._copy_images(train_rot, 'train', 'Rotten', food_name)
                    self._copy_images(val_rot, 'val', 'Rotten', food_name)
                    self._copy_images(test_rot, 'test', 'Rotten', food_name)
                    
                    total_images_processed += len(rotten_images)
        
        return total_images_processed > 0
        
    def _copy_images(self, images, split, label, food):
        """Copy images to destination with validation"""
        for img_path in images:
            filename = f"{food}_{img_path.name}"
            dst = self.output_dir / split / label / filename
            
            try:
                shutil.copy2(img_path, dst)
            except Exception as e:
                print(f"⚠️ Error copying {img_path}: {e}")
                
    def get_data_stats(self):
        """Print dataset statistics"""
        stats = {}
        
        print("\n" + "="*70)
        print("📊 DATASET STATISTICS".center(70))
        print("="*70)
        
        for split in ['train', 'val', 'test']:
            stats[split] = {}
            for label in ['Fresh', 'Slightly_Spoiled', 'Rotten']:
                path = self.output_dir / split / label
                count = len(list(path.iterdir())) if path.exists() else 0
                stats[split][label] = count
        
        # Print statistics
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            total = sum(stats[split].values())
            for label in ['Fresh', 'Slightly_Spoiled', 'Rotten']:
                count = stats[split][label]
                percentage = (count / total * 100) if total > 0 else 0
                bar = '█' * int(percentage / 2)
                print(f"  {label:20s}: {count:5d} ({percentage:5.1f}%) {bar}")
            print(f"  {'-'*60}")
            print(f"  {'TOTAL':20s}: {total:5d}")
        
        # Overall statistics
        total_all = sum(sum(stats[split].values()) for split in stats)
        print(f"\n{'='*70}")
        print(f"TOTAL IMAGES: {total_all}".center(70))
        print(f"{'='*70}\n")
        
        return stats
    
    def _print_download_instructions(self):
        """Print dataset download instructions"""
        print("\n" + "="*70)
        print("📥 DATASET DOWNLOAD INSTRUCTIONS".center(70))
        print("="*70)
        print("\nDataset: Fruit & Vegetable Fresh vs Rotten (5.21 GB)")
        print("URL: https://www.kaggle.com/datasets/zlatan599/fruitquality1")
        print("\n🔧 Setup Instructions:")
        print("1. Install Kaggle CLI:")
        print("   pip install kaggle")
        print("\n2. Get API token from: https://www.kaggle.com/settings/account")
        print("   - Click 'Create New API Token'")
        print("   - Download kaggle.json")
        print("\n3. Place kaggle.json in:")
        print("   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   Linux/Mac: ~/.kaggle/kaggle.json")
        print("\n4. Download dataset:")
        print("   kaggle datasets download -d zlatan599/fruitquality1")
        print("\n5. Extract dataset:")
        print(f"   Unzip to: {self.raw_dir}")
        print("\n6. Rerun this script:")
        print("   python preprocessing.py")
        print("="*70 + "\n")

def main():
    print("\n" + "="*70)
    print("🔥 FreshScanAI - DATA PREPROCESSING".center(70))
    print("="*70)
    
    # Check if raw data exists
    raw_dir = Path(DATA_CONFIG['raw_data_dir'])
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("\n⚠️ Raw data not found!")
        preprocessor = DataPreprocessor(DATA_CONFIG)
        preprocessor._print_download_instructions()
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(DATA_CONFIG)
    
    # Create directories
    preprocessor.create_directories()
    
    # Organize data
    print("\n🔄 Processing images...")
    success = preprocessor.organize_data_by_food()
    
    if success:
        # Print statistics
        stats = preprocessor.get_data_stats()
        print("✅ Data preprocessing completed successfully!")
        print("🎯 Ready for model training!")
    else:
        print("\n❌ Data preprocessing failed!")
        preprocessor._print_download_instructions()
        sys.exit(1)

if __name__ == "__main__":
    main()
