# 🚀 FreshScanAI - Quick Start Guide

Get your project running in minutes!

---

## ⚡ Quick Installation

### Step 1: Setup Environment
```powershell
# Navigate to project directory
cd "d:\btech\projects\Food spoilage detection\FreshScanAI"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset

#### Option A: Using Kaggle CLI (Recommended)
```powershell
# Install Kaggle
pip install kaggle

# Place kaggle.json in: C:\Users\<YourUsername>\.kaggle\

# Download dataset
kaggle datasets download -d zlatan599/fruitquality1

# Extract to data/raw/
Expand-Archive fruitquality1.zip -DestinationPath data/raw/
```

#### Option B: Manual Download
1. Visit: https://www.kaggle.com/datasets/zlatan599/fruitquality1
2. Click "Download" (requires Kaggle account)
3. Extract zip file to `data/raw/` folder

### Step 3: Preprocess Data
```powershell
python preprocessing.py
```
**Expected output**: Dataset split into train/val/test sets
**Time**: 5-10 minutes

### Step 4: Train Model
```powershell
python train_model.py
```
**Expected output**: Trained model saved to `models/freshscan_model.h5`
**Time**: 30-60 minutes (CPU) or 10-15 minutes (GPU)

### Step 5: Run Web App
```powershell
streamlit run app.py
```
**Access**: Open browser to http://localhost:8501

---

## 📋 Pre-Flight Checklist

Before running, ensure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted to `data/raw/`
- [ ] At least 8GB free disk space
- [ ] Internet connection (for first-time package downloads)

---

## 🔍 Verify Installation

Test if everything is installed correctly:

```powershell
# Check Python version
python --version  # Should be 3.8+

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"  # Should be 2.12+

# Check Streamlit
streamlit --version  # Should be 1.28+

# List installed packages
pip list
```

---

## 📂 Expected Directory Structure

After preprocessing, your structure should be:

```
FreshScanAI/
├── data/
│   ├── raw/                      # Downloaded dataset (5.21 GB)
│   │   ├── apple/
│   │   ├── banana/
│   │   └── ... (12 more)
│   └── processed/                # Preprocessed data
│       ├── train/
│       │   ├── Fresh/
│       │   ├── Slightly_Spoiled/
│       │   └── Rotten/
│       ├── val/
│       └── test/
├── models/
│   └── freshscan_model.h5       # Trained model (14 MB)
├── results/                      # Output visualizations
├── logs/                         # Training logs
└── ... (source files)
```

---

## 🎯 Usage Examples

### Example 1: Full Pipeline
```powershell
# 1. Preprocess data
python preprocessing.py

# 2. Train model
python train_model.py

# 3. Evaluate model
python evaluate.py

# 4. Test prediction
python predict.py

# 5. Launch web app
streamlit run app.py
```

### Example 2: Skip Training (Use Pre-trained Model)
If you have a pre-trained model:

```powershell
# Just run the web app
streamlit run app.py
```

### Example 3: Batch Prediction
```python
from predict import FreshScanAIPredictor

predictor = FreshScanAIPredictor()
results = predictor.batch_predict('path/to/images/')
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

- `IMG_SIZE`: Image dimensions (default: 224)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Maximum training epochs (default: 30)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)

---

## 🐛 Troubleshooting

### Issue 1: "Model file not found"
**Solution**: Train the model first with `python train_model.py`

### Issue 2: "Dataset not found"
**Solution**: Download and extract dataset to `data/raw/`

### Issue 3: TensorFlow not installing
**Solution**: 
```powershell
pip install --upgrade pip
pip install tensorflow==2.12.0 --no-cache-dir
```

### Issue 4: Streamlit port already in use
**Solution**:
```powershell
streamlit run app.py --server.port 8502
```

### Issue 5: Out of memory during training
**Solution**: Reduce batch size in `config.py`:
```python
TRAINING_CONFIG['batch_size'] = 16  # Instead of 32
```

### Issue 6: Slow training on CPU
**Solution**: This is normal. Options:
1. Use Google Colab (free GPU)
2. Reduce epochs to 15-20
3. Use smaller model

---

## 💡 Tips for Best Results

### For Training:
1. Use GPU if available (10x faster)
2. Monitor training logs in real-time
3. Stop if validation accuracy plateaus
4. Save multiple checkpoints

### For Inference:
1. Use well-lit, clear images
2. Capture entire food item
3. Avoid shadows and reflections
4. Use neutral background

### For Presentation:
1. Prepare sample images beforehand
2. Test app before demo
3. Have backup screenshots/video
4. Explain results clearly

---

## 🚀 Deployment Options

### Option 1: Local Deployment
```powershell
streamlit run app.py
```
Access: http://localhost:8501

### Option 2: Streamlit Cloud (Free)
1. Push code to GitHub
2. Visit: https://streamlit.io/cloud
3. Connect repository
4. Deploy!

### Option 3: Docker (Advanced)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Training Time (CPU) | 30-60 minutes |
| Training Time (GPU) | 10-15 minutes |
| Model Size | 14 MB |
| Inference Time | <1 second |
| Accuracy | 92-95% |
| Memory Usage | ~2 GB |

---

## 🎓 For Academic Submission

### What to Submit:

1. **Source Code**
   - All Python files
   - Configuration files
   - Requirements.txt

2. **Documentation**
   - README.md
   - ABSTRACT.md
   - Report (PDF)

3. **Results**
   - Confusion matrix
   - Training history plots
   - Classification report
   - Sample predictions

4. **Presentation**
   - PowerPoint slides
   - Demo video (optional)

5. **Model** (Optional)
   - Trained model file (.h5)
   - Or Google Drive link

### Submission Checklist:

- [ ] Code is well-commented
- [ ] README has clear instructions
- [ ] All documentation is complete
- [ ] Results folder has all plots
- [ ] Presentation is ready
- [ ] Demo tested and working
- [ ] Report follows guidelines
- [ ] References are cited

---

## 📞 Support & Resources

### Documentation:
- Main README: `README.md`
- Abstract: `documentation/ABSTRACT.md`
- Presentation: `documentation/PRESENTATION.md`
- Viva Q&A: `documentation/VIVA_QA.md`

### External Resources:
- TensorFlow Docs: https://tensorflow.org/tutorials
- Streamlit Docs: https://docs.streamlit.io
- Dataset: https://kaggle.com/datasets/zlatan599/fruitquality1
- MobileNetV2 Paper: https://arxiv.org/abs/1801.04381

### Contact:
- GitHub Issues: (your repo)
- Email: your.email@example.com

---

## 🎉 Success Criteria

Your project is ready when:

✅ Data is preprocessed successfully  
✅ Model achieves >90% accuracy  
✅ Web app runs without errors  
✅ Can make predictions on test images  
✅ Results are visualized properly  
✅ Documentation is complete  
✅ Presentation is prepared  
✅ You can explain everything in viva!

---

## 🏆 Next Steps

After basic setup:

1. **Experiment**: Try different hyperparameters
2. **Improve**: Add more food categories
3. **Extend**: Build mobile app
4. **Deploy**: Host on cloud platform
5. **Publish**: Share on GitHub
6. **Present**: Showcase at college event

---

**Ready to build something amazing? Let's go! 🚀**

**Good Luck with Your Project! 🎓**
