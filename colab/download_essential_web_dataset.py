#!/usr/bin/env python3
"""
Download Essential Web v1.0 Dataset from Hugging Face
A high-quality web text dataset for language model training
"""

import os
import pandas as pd
from datasets import load_dataset
import requests
from pathlib import Path

def download_essential_web_dataset():
    """
    Download the Essential Web v1.0 dataset from Hugging Face
    Dataset: https://huggingface.co/datasets/sumuks/essential-web-v1.0-sample-10M
    """
    
    print("🌐 Downloading Essential Web v1.0 Dataset...")
    print("📊 Dataset Info: 10M tokens of high-quality web text")
    print("🔗 Source: https://huggingface.co/datasets/sumuks/essential-web-v1.0-sample-10M")
    
    try:
        # Method 1: Using datasets library (recommended)
        print("\n📦 Loading dataset using HuggingFace datasets library...")
        dataset = load_dataset("sumuks/essential-web-v1.0-sample-10M", split="train")
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Save as parquet file
        output_file = "essential_web_v1_sample_10M.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"✅ Dataset downloaded and saved as: {output_file}")
        print(f"📋 Dataset shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")
        
        # Show sample data
        print("\n🔍 Sample data:")
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                sample_text = str(df[col].iloc[0])[:100] + "..." if len(str(df[col].iloc[0])) > 100 else str(df[col].iloc[0])
                print(f"  {col}: {sample_text}")
        
        return output_file, df
        
    except Exception as e:
        print(f"❌ Error with datasets library: {e}")
        print("🔄 Trying direct download method...")
        
        # Method 2: Direct download (fallback)
        return download_direct()

def download_direct():
    """Direct download method as fallback"""
    
    # Direct parquet file URL
    parquet_url = "https://huggingface.co/datasets/sumuks/essential-web-v1.0-sample-10M/resolve/main/data/part-00000.parquet"
    output_file = "essential_web_v1_sample_10M.parquet"
    
    print(f"📥 Downloading directly from: {parquet_url}")
    
    try:
        response = requests.get(parquet_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)", end='', flush=True)
        
        print(f"\n✅ Downloaded: {output_file}")
        
        # Load and inspect the parquet file
        df = pd.read_parquet(output_file)
        print(f"📋 Dataset shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")
        
        return output_file, df
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None, None

def inspect_dataset(df):
    """Inspect the dataset structure and content"""
    
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    print(f"📏 Total rows: {len(df):,}")
    print(f"📝 Columns: {len(df.columns)}")
    
    for col in df.columns:
        print(f"\n🔍 Column: '{col}'")
        print(f"   Type: {df[col].dtype}")
        print(f"   Non-null: {df[col].count():,} ({df[col].count()/len(df)*100:.1f}%)")
        
        if df[col].dtype == 'object':  # Text column
            # Calculate text statistics
            text_lengths = df[col].dropna().astype(str).str.len()
            total_chars = text_lengths.sum()
            
            print(f"   Total characters: {total_chars:,}")
            print(f"   Avg length: {text_lengths.mean():.0f} chars")
            print(f"   Max length: {text_lengths.max():,} chars")
            
            # Show sample
            sample = str(df[col].dropna().iloc[0])
            if len(sample) > 200:
                sample = sample[:200] + "..."
            print(f"   Sample: {sample}")

def suggest_training_config(df):
    """Suggest optimal training configuration based on dataset"""
    
    print("\n" + "="*60)
    print("🎯 TRAINING RECOMMENDATIONS")
    print("="*60)
    
    # Find text column
    text_column = None
    for col in df.columns:
        if df[col].dtype == 'object':
            if col.lower() in ['text', 'content', 'body', 'article', 'document']:
                text_column = col
                break
    
    if text_column is None:
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_column = text_columns[0]
    
    if text_column:
        # Calculate dataset size
        text_data = df[text_column].dropna().astype(str)
        total_chars = text_data.str.len().sum()
        unique_chars = len(set(''.join(text_data.head(1000))))  # Sample for unique chars
        
        print(f"📝 Recommended text column: '{text_column}'")
        print(f"📊 Total characters: {total_chars:,}")
        print(f"🔤 Estimated vocab size: ~{unique_chars} characters")
        
        # Model size recommendations
        print("\n🎛️ Recommended model configurations:")
        
        if total_chars < 1_000_000:  # < 1M chars
            print("   🔹 Small model: --d-model 256 --n-layer 4 --epochs 15")
        elif total_chars < 10_000_000:  # < 10M chars  
            print("   🔹 Medium model: --d-model 512 --n-layer 6 --epochs 12")
        else:  # 10M+ chars
            print("   🔹 Large model: --d-model 768 --n-layer 8 --epochs 10")
        
        # Training command
        print(f"\n🚀 Training command:")
        print(f"python train_with_datasets.py --dataset custom --custom-file essential_web_v1_sample_10M.parquet --text-column {text_column} --epochs 12")
        
    else:
        print("❌ No suitable text column found!")

def main():
    """Main function to download and analyze the dataset"""
    
    print("🌐 Essential Web v1.0 Dataset Downloader")
    print("="*50)
    
    # Download dataset
    parquet_file, df = download_essential_web_dataset()
    
    if parquet_file and df is not None:
        # Analyze dataset
        inspect_dataset(df)
        
        # Training suggestions
        suggest_training_config(df)
        
        print(f"\n✅ Ready to train! Dataset saved as: {parquet_file}")
        print("\n📚 Next steps:")
        print("1. Use the training command above")
        print("2. Or run: python dataset_examples.py (for interactive menu)")
        print("3. Monitor training progress in TensorBoard logs")
        
    else:
        print("❌ Failed to download dataset. Please check your internet connection.")

if __name__ == "__main__":
    # Install required packages (for Colab)
    print("📦 Installing required packages...")
    os.system("pip install datasets pandas pyarrow")
    
    # Run main function
    main() 