import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import requests
import sys
import os
import argparse
from pathlib import Path
import pandas as pd

# Handle different environments (local vs Colab)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths for local development
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(current_dir, 'mamba_tiny_master'))

# Add paths for Colab environment
if '/content' in os.getcwd():
    sys.path.append('/content')
    sys.path.append('/content/babyBhasha')
    sys.path.append('/content/babyBhasha/colab')
    sys.path.append('/content/babyBhasha/colab/mamba_tiny_master')

from mamba_tiny_master.model import Mamba, ModelArgs

class FlexibleDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="shakespeare", batch_size=32, seq_len=128, custom_file=None, text_column=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.custom_file = custom_file
        self.text_column = text_column  # For parquet files, specify which column contains text
        self.tokenizer = None
        self.vocab_size = None
        self.dataset = None
        
        # Available datasets
        self.datasets = {
            "shakespeare": {
                "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                "description": "Tiny Shakespeare - Classic character-level dataset"
            },
            "leo_tolstoy": {
                "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                "description": "Leo Tolstoy works"
            },
            "poetry": {
                "url": "https://raw.githubusercontent.com/aparrish/gutenberg-poetry-corpus/master/quick-test.txt",
                "description": "Poetry corpus from Project Gutenberg"
            },
            "bible": {
                "url": "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/English-WEB.txt",
                "description": "King James Bible text"
            },
            "nietzsche": {
                "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/nietzsche/input.txt", 
                "description": "Friedrich Nietzsche writings"
            },
            "linux_kernel": {
                "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux/input.txt",
                "description": "Linux kernel source code"
            },
            "custom": {
                "url": None,
                "description": "Custom text file (.txt) or Parquet file (.parquet) provided by user"
            }
        }

    def prepare_data(self):
        """
        Downloads and prepares data, builds vocabulary.
        This is called once per node and prepares the data.
        """
        if self.dataset_name == "custom" and self.custom_file:
            text = self.load_custom_file(self.custom_file)
        elif self.dataset_name in self.datasets and self.datasets[self.dataset_name]["url"]:
            try:
                print(f"Downloading {self.datasets[self.dataset_name]['description']}...")
                url = self.datasets[self.dataset_name]["url"]
                response = requests.get(url)
                response.raise_for_status()
                text = response.text
                print(f"Downloaded {len(text)} characters")
            except Exception as e:
                print(f"Failed to download {self.dataset_name}: {e}")
                print("Falling back to Shakespeare...")
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                response = requests.get(url)
                text = response.text
        else:
            print(f"Unknown dataset: {self.dataset_name}")
            print("Available datasets:", list(self.datasets.keys()))
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Build vocabulary from characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Character set: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
        
        # Create character-level tokenizer
        self.tokenizer = {ch: i for i, ch in enumerate(chars)}
        self.decode = {i: ch for i, ch in enumerate(chars)}

    def load_custom_file(self, file_path):
        """Load custom file - supports .txt and .parquet formats"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.parquet':
            try:
                print(f"Loading Parquet file: {file_path}")
                df = pd.read_parquet(file_path)
                print(f"Parquet file loaded with {len(df)} rows and columns: {list(df.columns)}")
                
                # Auto-detect text column if not specified
                if self.text_column is None:
                    # Common text column names
                    text_candidates = ['text', 'content', 'body', 'message', 'description', 'story', 'article']
                    
                    for col in text_candidates:
                        if col in df.columns:
                            self.text_column = col
                            break
                    
                    # If no common names found, use first string column
                    if self.text_column is None:
                        string_cols = df.select_dtypes(include=['object', 'string']).columns
                        if len(string_cols) > 0:
                            self.text_column = string_cols[0]
                        else:
                            raise ValueError("No text columns found in Parquet file. Please specify --text-column")
                
                print(f"Using text column: '{self.text_column}'")
                
                if self.text_column not in df.columns:
                    raise ValueError(f"Column '{self.text_column}' not found. Available columns: {list(df.columns)}")
                
                # Extract text and concatenate
                text_series = df[self.text_column].dropna()
                text = '\n'.join(text_series.astype(str))
                print(f"Extracted {len(text)} characters from {len(text_series)} rows")
                
                return text
                
            except Exception as e:
                print(f"Error loading Parquet file: {e}")
                raise
                
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Loaded text file: {file_path}")
                print(f"File size: {len(text)} characters")
                return text
            except Exception as e:
                print(f"Error loading text file: {e}")
                raise
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .txt, .parquet")

    def setup(self, stage=None):
        """
        Setup method called on each process to create the actual dataset.
        This happens after prepare_data() and creates training examples.
        """
        # Load the text data
        if self.dataset_name == "custom" and self.custom_file:
            text = self.load_custom_file(self.custom_file)
        elif self.dataset_name in self.datasets and self.datasets[self.dataset_name]["url"]:
            try:
                print(f"Downloading {self.datasets[self.dataset_name]['description']}...")
                url = self.datasets[self.dataset_name]["url"]
                response = requests.get(url)
                response.raise_for_status()
                text = response.text
                print(f"Downloaded {len(text)} characters")
            except Exception as e:
                print(f"Failed to download {self.dataset_name}: {e}")
                print("Falling back to Shakespeare...")
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                response = requests.get(url)
                text = response.text
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # If vocabulary hasn't been created yet, create it
        if self.tokenizer is None:
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            print(f"Vocabulary size: {self.vocab_size}")
            print(f"Character set: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
            
            # Create character-level tokenizer
            self.tokenizer = {ch: i for i, ch in enumerate(chars)}
            self.decode = {i: ch for i, ch in enumerate(chars)}

        # Tokenize the entire text
        tokenized_ids = [self.tokenizer[c] for c in text if c in self.tokenizer]

        # Create training examples by sliding window
        examples = []
        for i in range(0, len(tokenized_ids) - self.seq_len, self.seq_len):
            examples.append(torch.tensor(tokenized_ids[i:i + self.seq_len], dtype=torch.long))
            
        self.dataset = examples
        print(f"Created {len(examples)} training examples")

    def train_dataloader(self):
        # Detect if we're on TPU or have limited memory
        try:
            import torch_xla
            is_tpu = True
        except ImportError:
            is_tpu = False
        
        # Use no workers for TPU or CPU-only environments to avoid memory issues
        num_workers = 0 if (is_tpu or not torch.cuda.is_available()) else 1
        pin_memory = torch.cuda.is_available() and not is_tpu
        
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )

    def list_available_datasets(self):
        """List all available datasets"""
        print("\n📚 Available Datasets:")
        print("=" * 60)
        for name, info in self.datasets.items():
            print(f"  {name:15} - {info['description']}")
        print("=" * 60)

class MambaLightningModule(pl.LightningModule):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.model = Mamba(model_args)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        input_ids = batch
        logits = self.model(input_ids)
        
        # Shift logits and labels for next token prediction
        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def on_train_start(self):
        # Log number of parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log('num_params', float(num_params))
        print(f"Model has {num_params/1e6:.2f}M parameters")

def main():
    parser = argparse.ArgumentParser(description='Train Mamba model on various datasets')
    parser.add_argument('--dataset', type=str, default='shakespeare', 
                       help='Dataset to use (shakespeare, leo_tolstoy, poetry, bible, nietzsche, linux_kernel, custom)')
    parser.add_argument('--custom-file', type=str, help='Path to custom text file (required if dataset=custom)')
    parser.add_argument('--text-column', type=str, help='For custom Parquet files, specify the column name containing text')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layer', type=int, default=8, help='Number of layers')
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')
    
    args = parser.parse_args()
    
    # Create data module to list datasets if requested
    data_module = FlexibleDataModule()
    if args.list_datasets:
        data_module.list_available_datasets()
        return
    
    # Validate custom file requirement
    if args.dataset == 'custom' and not args.custom_file:
        print("❌ Error: --custom-file is required when using dataset='custom'")
        return
    
    if args.custom_file and not os.path.exists(args.custom_file):
        print(f"❌ Error: Custom file not found: {args.custom_file}")
        return

    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Setup data module
    data_module = FlexibleDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        custom_file=args.custom_file,
        text_column=args.text_column
    )
    data_module.prepare_data()
    data_module.setup() # Call setup after prepare_data

    # Model arguments
    model_args = ModelArgs(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=data_module.vocab_size, 
        expand=args.expand,
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'mamba-{args.dataset}-{{epoch:02d}}-{{train_loss:.2f}}',
        save_top_k=3,
        monitor='train_loss'
    )

    # Create model
    model = MambaLightningModule(model_args)

    # Configure trainer with TPU-safe settings
    try:
        import torch_xla
        is_tpu = True
    except ImportError:
        is_tpu = False
    
    trainer_args = {
        'max_epochs': args.epochs,
        'callbacks': [checkpoint_callback],
        'default_root_dir': 'logs',
        'enable_progress_bar': True,
        'log_every_n_steps': 50,
    }
    
    if is_tpu:
        # TPU-specific configuration
        trainer_args.update({
            'accelerator': 'tpu',
            'devices': 'auto',
            'precision': 32,  # Use 32-bit for TPU stability
        })
    else:
        # GPU/CPU configuration
        trainer_args.update({
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': '16-mixed' if torch.cuda.is_available() else 32,
        })
    
    trainer = pl.Trainer(**trainer_args)
    
    # Start training
    print(f"\n🚀 Starting training with {args.dataset} dataset...")
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)
    
    # Save the final model with dataset name
    final_checkpoint = f"mamba_{args.dataset}_final.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"✅ Final model saved as: {final_checkpoint}")
    
    # Save the tokenizer with dataset name
    import pickle
    tokenizer_file = f'tokenizer_{args.dataset}.pkl'
    decode_file = f'decode_{args.dataset}.pkl'
    
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(data_module.tokenizer, f)
    with open(decode_file, 'wb') as f:
        pickle.dump(data_module.decode, f)
    
    print(f"✅ Tokenizer saved as: {tokenizer_file}")
    print(f"✅ Decoder saved as: {decode_file}")
    
    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Training completed!")
    print(f"   Model parameters: {num_params/1e6:.2f}M")
    print(f"   Dataset: {args.dataset}")
    print(f"   Vocabulary size: {data_module.vocab_size}")

if __name__ == '__main__':
    main() 