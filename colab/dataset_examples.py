#!/usr/bin/env python3
"""
Dataset Examples for Mamba Training
Easy commands to train your Mamba model on different datasets
"""

import os
import subprocess
import glob

def print_banner():
    print("\n" + "="*70)
    print("🎭 MAMBA DATASET TRAINING - Multiple Dataset Options")
    print("="*70)

def list_available_datasets():
    """Show all available datasets"""
    datasets = {
        "shakespeare": "Classic Shakespeare plays - great for literary text",
        "nietzsche": "Friedrich Nietzsche philosophical writings",
        "bible": "King James Bible - religious/classical text",
        "poetry": "Poetry corpus from Project Gutenberg",
        "linux_kernel": "Linux kernel source code - for code generation",
        "custom": "Your own text file (.txt) or Parquet file (.parquet)"
    }
    
    print("\n📚 Available Datasets:")
    print("-" * 50)
    for name, desc in datasets.items():
        print(f"  {name:12} - {desc}")
    print("-" * 50)
    print("\n💡 Custom Dataset Support:")
    print("  📄 Text files (.txt): Direct text content")
    print("  📊 Parquet files (.parquet): Tabular data with text columns")
    print("     - Auto-detects common text columns: 'text', 'content', 'body', etc.")
    print("     - Or specify column with --text-column parameter")

def show_example_commands():
    """Show example training commands"""
    print("\n⚡ Example Training Commands:")
    print("-" * 50)
    
    examples = [
        ("Shakespeare (default)", "python train_with_datasets.py --dataset shakespeare --epochs 10"),
        ("Nietzsche Philosophy", "python train_with_datasets.py --dataset nietzsche --epochs 15"),
        ("Bible Text", "python train_with_datasets.py --dataset bible --epochs 12"),
        ("Poetry Corpus", "python train_with_datasets.py --dataset poetry --epochs 10"),
        ("Linux Kernel Code", "python train_with_datasets.py --dataset linux_kernel --epochs 8"),
        ("Custom Text File", "python train_with_datasets.py --dataset custom --custom-file your_file.txt"),
        ("Custom Parquet File", "python train_with_datasets.py --dataset custom --custom-file train-00000-of-03291.parquet"),
        ("Parquet with Specific Column", "python train_with_datasets.py --dataset custom --custom-file data.parquet --text-column content"),
        ("Small Model (256K params)", "python train_with_datasets.py --dataset shakespeare --d-model 256 --n-layer 6"),
        ("Large Model (2M params)", "python train_with_datasets.py --dataset shakespeare --d-model 768 --n-layer 10"),
    ]
    
    for desc, cmd in examples:
        print(f"\n  {desc}:")
        print(f"    {cmd}")
    
    print("\n💡 Pro Tips:")
    print("  - Use --list-datasets to see all options")
    print("  - Models are saved as 'mamba_{dataset}_final.ckpt'")
    print("  - Each dataset gets its own tokenizer files")
    print("  - Use --resume to continue training from checkpoint")
    print("  - For Parquet files, it auto-detects text columns or use --text-column")

def quick_train_menu():
    """Interactive menu for quick training"""
    print("\n🚀 Quick Training Menu:")
    print("=" * 30)
    
    options = [
        ("1", "Train on Shakespeare (classic)", "shakespeare"),
        ("2", "Train on Nietzsche (philosophy)", "nietzsche"), 
        ("3", "Train on Bible (religious text)", "bible"),
        ("4", "Train on Poetry", "poetry"),
        ("5", "Train on Linux Code", "linux_kernel"),
        ("6", "Train on custom file (auto-detect)", "custom"),
        ("q", "Quit", None)
    ]
    
    for key, desc, _ in options:
        print(f"  {key}. {desc}")
    
    choice = input("\nEnter your choice (1-6, q): ").strip().lower()
    
    # Find selected dataset
    selected = None
    for key, desc, dataset in options:
        if choice == key:
            selected = dataset
            break
    
    if choice == 'q':
        print("👋 Goodbye!")
        return
    
    if selected is None:
        print("❌ Invalid choice!")
        return
    
    # Build command
    cmd = ["python", "train_with_datasets.py", "--dataset", selected]
    
    if selected == "custom":
        # Auto-detect available custom files
        custom_files = []
        
        # Check for common file patterns
        patterns = [
            "*.txt",
            "*.parquet", 
            "test.txt",
            "train-*.parquet",
            "train-00000-of-*.parquet"
        ]
        
        for pattern in patterns:
            custom_files.extend(glob.glob(pattern))
        
        # Remove duplicates and sort
        custom_files = sorted(list(set(custom_files)))
        
        if not custom_files:
            print("❌ No custom files found! Looking for .txt or .parquet files.")
            return
        
        print(f"\n📁 Found {len(custom_files)} custom file(s):")
        for i, file in enumerate(custom_files, 1):
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"  {i}. {file} ({file_size:.1f} MB)")
        
        if len(custom_files) == 1:
            selected_file = custom_files[0]
            print(f"🎯 Using: {selected_file}")
        else:
            try:
                file_choice = int(input(f"\nSelect file (1-{len(custom_files)}): ")) - 1
                if 0 <= file_choice < len(custom_files):
                    selected_file = custom_files[file_choice]
                else:
                    print("❌ Invalid choice!")
                    return
            except ValueError:
                print("❌ Invalid input!")
                return
        
        cmd.extend(["--custom-file", selected_file])
        
        # If it's a parquet file, ask about text column
        if selected_file.endswith('.parquet'):
            print("\n📊 Parquet file detected!")
            print("Common text columns: text, content, body, message, description")
            text_col = input("Text column name (leave empty for auto-detection): ").strip()
            if text_col:
                cmd.extend(["--text-column", text_col])
    
    # Ask for model size
    print("\n🔧 Model Size Options:")
    print("  1. Small (256K params) - Fast training")
    print("  2. Medium (1M params) - Balanced")
    print("  3. Large (2M params) - Better quality")
    
    size_choice = input("Model size (1-3, default=2): ").strip() or "2"
    
    if size_choice == "1":
        cmd.extend(["--d-model", "256", "--n-layer", "6"])
        print("🔹 Using small model configuration")
    elif size_choice == "3":
        cmd.extend(["--d-model", "768", "--n-layer", "10"])
        print("🔹 Using large model configuration")
    else:
        cmd.extend(["--d-model", "512", "--n-layer", "8"])
        print("🔹 Using medium model configuration")
    
    # Ask for epochs
    epochs = input("Number of epochs (default=10): ").strip() or "10"
    cmd.extend(["--epochs", epochs])
    
    print(f"\n🚀 Starting training with command:")
    print(f"   {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop training anytime...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted!")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print_banner()
    list_available_datasets()
    show_example_commands()
    
    if input("\n🎯 Would you like to start quick training? (y/n): ").strip().lower().startswith('y'):
        quick_train_menu()
    else:
        print("\n📖 Use the example commands above to train on your preferred dataset!")
        print("💡 Run: python train_with_datasets.py --list-datasets for more info")

if __name__ == "__main__":
    main() 