#!/usr/bin/env python3
"""
Google Colab Setup Script for Mamba Training with Essential Web Dataset
Run this first in your Colab notebook to set everything up
"""

import os
import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    
    print("📦 Installing required packages...")
    
    packages = [
        "torch",
        "pytorch-lightning", 
        "datasets",
        "pandas",
        "pyarrow",
        "transformers",
        "einops",
        "mamba-ssm",
        "causal-conv1d>=1.0.0",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      capture_output=True)
    
    print("✅ All packages installed!")

def setup_repository():
    """Clone the repository and set up the environment"""
    
    print("📂 Setting up repository...")
    
    # Clone repository (if not already cloned)
    if not os.path.exists("tinymamba"):
        subprocess.run(["git", "clone", "https://github.com/your-repo/tinymamba.git"], 
                      capture_output=True)
    
    # Change to colab directory
    os.chdir("tinymamba/colab")
    
    print("✅ Repository setup complete!")

def download_essential_web():
    """Download and analyze the Essential Web dataset"""
    
    print("🌐 Downloading Essential Web v1.0 Dataset...")
    
    # Run the download script
    exec(open("download_essential_web_dataset.py").read())

def show_quick_start_guide():
    """Show quick start commands for training"""
    
    print("\n" + "="*70)
    print("🚀 GOOGLE COLAB QUICK START GUIDE")
    print("="*70)
    
    print("\n📖 Available Training Options:")
    print("-" * 40)
    
    commands = [
        ("📊 Train on Essential Web Dataset", 
         "!python train_with_datasets.py --dataset custom --custom-file essential_web_v1_sample_10M.parquet --epochs 12"),
        
        ("🎭 Resume Shakespeare Training", 
         "!python resume_training.py --additional-epochs 10"),
        
        ("📚 Train on Bible Text", 
         "!python train_with_datasets.py --dataset bible --epochs 15"),
        
        ("🧠 Train on Nietzsche Philosophy", 
         "!python train_with_datasets.py --dataset nietzsche --epochs 12"),
        
        ("💻 Train on Linux Kernel Code", 
         "!python train_with_datasets.py --dataset linux_kernel --epochs 10"),
        
        ("🎯 Interactive Dataset Menu", 
         "!python dataset_examples.py"),
        
        ("💬 Chat with Trained Model", 
         "!python chat.py"),
    ]
    
    for desc, cmd in commands:
        print(f"\n{desc}:")
        print(f"  {cmd}")
    
    print("\n💡 Pro Tips for Colab:")
    print("- Add '!' before python commands in notebook cells")
    print("- Use Runtime > Change runtime type > GPU for faster training")
    print("- Download your models before session expires:")
    print("  from google.colab import files")
    print("  files.download('mamba_custom_final.ckpt')")
    
    print("\n🔧 Model Size Recommendations:")
    print("- Small (256K params): Fast training, good for testing")
    print("- Medium (1M params): Balanced performance and quality")  
    print("- Large (2M params): Best quality, slower training")

def main():
    """Main setup function"""
    
    print("🎯 Mamba Language Model Training Setup for Google Colab")
    print("="*60)
    
    try:
        # Step 1: Install packages
        install_requirements()
        
        # Step 2: Download dataset
        download_essential_web()
        
        # Step 3: Show guide
        show_quick_start_guide()
        
        print("\n✅ Setup Complete! You're ready to train your Mamba model!")
        print("🎉 Choose any command above to start training.")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("💡 Try running individual steps manually.")

if __name__ == "__main__":
    main()


# ============================================================================
# GOOGLE COLAB CELL EXAMPLES
# Copy these into separate cells in your Colab notebook
# ============================================================================

"""
CELL 1: Setup (Run this first)
```python
!wget https://raw.githubusercontent.com/your-repo/tinymamba/main/colab/colab_setup.py
exec(open('colab_setup.py').read())
```

CELL 2: Train on Essential Web Dataset  
```python
!python train_with_datasets.py --dataset custom --custom-file essential_web_v1_sample_10M.parquet --epochs 12
```

CELL 3: Chat with your model
```python
!python chat.py
```

CELL 4: Download your trained model
```python
from google.colab import files
files.download('mamba_custom_final.ckpt')
files.download('tokenizer_custom.pkl') 
files.download('decode_custom.pkl')
```
""" 