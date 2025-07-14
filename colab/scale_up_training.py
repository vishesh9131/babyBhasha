import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import argparse
import pickle
import glob
from pathlib import Path
import pandas as pd
import requests

import sys

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
from train_with_datasets import FlexibleDataModule, MambaLightningModule

class ScaledMambaLightningModule(pl.LightningModule):
    """
    Custom Lightning module for scaled training with adjustable learning rate
    """
    def __init__(self, model_args: ModelArgs, learning_rate: float = 5e-4):
        super().__init__()
        self.model_args = model_args
        self.model = Mamba(model_args)
        self.learning_rate = learning_rate
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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_start(self):
        # Log number of parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log('num_params', float(num_params))
        print(f"Model has {num_params/1e6:.2f}M parameters")

class FlexibleDataModuleWithColumnDrop(FlexibleDataModule):
    """
    Extended data module that can drop the first column from parquet files
    """
    def __init__(self, dataset_name="shakespeare", batch_size=32, seq_len=128, 
                 custom_file=None, text_column=None, drop_first_column=False):
        super().__init__(dataset_name, batch_size, seq_len, custom_file, text_column)
        self.drop_first_column = drop_first_column

    def load_custom_file(self, file_path):
        """Load custom file - supports .txt and .parquet formats with optional column dropping"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.parquet':
            try:
                print(f"Loading Parquet file: {file_path}")
                df = pd.read_parquet(file_path)
                print(f"Original Parquet file: {len(df)} rows and columns: {list(df.columns)}")
                
                # Drop first column if requested
                if self.drop_first_column and len(df.columns) > 1:
                    first_col = df.columns[0]
                    df = df.drop(columns=[first_col])
                    print(f"✂️ Dropped first column '{first_col}'. Remaining columns: {list(df.columns)}")
                
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
        Setup method that properly handles column dropping for parquet files
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

def transfer_compatible_weights(small_model, large_model):
    """
    Transfer compatible weights from smaller model to larger model.
    Only transfers layers that have exact size matches.
    """
    small_state = small_model.state_dict()
    large_state = large_model.state_dict()
    
    transferred_layers = []
    skipped_layers = []
    
    for name, param in small_state.items():
        if name in large_state:
            if param.shape == large_state[name].shape:
                large_state[name] = param.clone()
                transferred_layers.append(name)
            else:
                skipped_layers.append(f"{name}: {param.shape} -> {large_state[name].shape}")
        else:
            skipped_layers.append(f"{name}: not found in large model")
    
    large_model.load_state_dict(large_state)
    
    print(f"✅ Transferred {len(transferred_layers)} compatible layers")
    print(f"⚠️  Skipped {len(skipped_layers)} incompatible layers")
    
    if len(skipped_layers) <= 10:  # Show details if not too many
        print("\n🔍 Skipped layers:")
        for layer in skipped_layers:
            print(f"   {layer}")
    
    return transferred_layers, skipped_layers

def main():
    parser = argparse.ArgumentParser(description='Scale up Mamba model and continue training')
    
    # Model scaling options
    parser.add_argument('--source-checkpoint', type=str, required=True,
                       help='Path to source checkpoint to transfer from')
    parser.add_argument('--source-dataset', type=str, default='custom',
                       help='Dataset used for source model (for tokenizer loading)')
    parser.add_argument('--train-from-scratch', action='store_true',
                       help='Train from scratch instead of transfer learning (ignores source checkpoint)')
    
    # New model configuration
    parser.add_argument('--d-model', type=int, default=768, 
                       help='New model dimension (default: 768 for ~25M params)')
    parser.add_argument('--n-layer', type=int, default=12,
                       help='New number of layers (default: 12)')
    parser.add_argument('--expand', type=int, default=2,
                       help='Expansion factor (default: 2)')
    
    # Training configuration
    parser.add_argument('--dataset', type=str, default='custom',
                       help='Dataset to continue training on')
    parser.add_argument('--custom-file', type=str,
                       help='Path to custom file (if dataset=custom)')
    parser.add_argument('--text-column', type=str,
                       help='Text column for parquet files')
    parser.add_argument('--drop-first-column', action='store_true',
                       help='Drop the first column from parquet files before processing')
    parser.add_argument('--force-tpu', action='store_true',
                       help='Force TPU usage, bypass TPU detection (useful in Colab)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Additional epochs to train')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (reduced for larger model)')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (reduced for fine-tuning)')
    
    # Output configuration
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_scaled',
                       help='Output checkpoint directory')
    parser.add_argument('--output-name', type=str, default='scaled',
                       help='Output model name suffix')
    
    args = parser.parse_args()
    
    print("🔄 Mamba Model Scale-Up Training")
    print("=" * 50)
    
    # Initialize variables for source model info
    source_model = None
    source_args = None
    source_params = 0
    
    # Handle source checkpoint loading or training from scratch
    if args.train_from_scratch:
        print("🆕 Training from scratch (no transfer learning)")
        print(f"📂 Target model: {args.d_model}D, {args.n_layer}L")
    else:
        # Check source checkpoint
        if not os.path.exists(args.source_checkpoint):
            print(f"❌ Source checkpoint not found: {args.source_checkpoint}")
            print("💡 Consider using --train-from-scratch to train without transfer learning")
            return
        
        print(f"📂 Source checkpoint: {args.source_checkpoint}")
        
        # Load source model to get configuration
        try:
            source_lightning = MambaLightningModule.load_from_checkpoint(args.source_checkpoint)
            source_model = source_lightning.model
            source_args = source_model.args
            
            print(f"📊 Source model: {source_args.d_model}D, {source_args.n_layer}L")
            source_params = sum(p.numel() for p in source_model.parameters())
            print(f"   Parameters: {source_params/1e6:.1f}M")
            
        except Exception as e:
            print(f"❌ Error loading source model: {e}")
            print("💡 Consider using --train-from-scratch to train without transfer learning")
            return
    
    # Setup data module with column dropping option
    data_module = FlexibleDataModuleWithColumnDrop(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        custom_file=args.custom_file,
        text_column=args.text_column,
        drop_first_column=args.drop_first_column
    )
    
    # Load existing tokenizer if available
    tokenizer_file = f'tokenizer_{args.source_dataset}.pkl'
    decode_file = f'decode_{args.source_dataset}.pkl'
    
    if os.path.exists(tokenizer_file) and os.path.exists(decode_file):
        print(f"📝 Loading existing tokenizer: {tokenizer_file}")
        with open(tokenizer_file, 'rb') as f:
            data_module.tokenizer = pickle.load(f)
        with open(decode_file, 'rb') as f:
            data_module.decode = pickle.load(f)
        data_module.vocab_size = len(data_module.tokenizer)
        print(f"   Vocabulary size: {data_module.vocab_size}")
    else:
        print("📝 Creating new vocabulary from dataset...")
        data_module.prepare_data()
    
    # Create larger model configuration
    new_model_args = ModelArgs(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=data_module.vocab_size,
        expand=args.expand,
    )
    
    print(f"\n🔬 New model: {new_model_args.d_model}D, {new_model_args.n_layer}L")
    
    # Create new model
    new_model = Mamba(new_model_args)
    new_params = sum(p.numel() for p in new_model.parameters())
    
    if args.train_from_scratch:
        print(f"   Parameters: {new_params/1e6:.1f}M (training from scratch)")
        print(f"\n🆕 Initializing new model with random weights...")
        transferred, skipped = [], list(new_model.state_dict().keys())
        print(f"✅ Model initialized with {len(skipped)} randomly initialized layers")
    else:
        print(f"   Parameters: {new_params/1e6:.1f}M ({new_params/source_params:.1f}x larger)")
        
        # Transfer compatible weights
        print(f"\n🔄 Transferring weights from source model...")
        transferred, skipped = transfer_compatible_weights(source_model, new_model)
    
    # Create new Lightning module with appropriate learning rate
    # Use higher learning rate for training from scratch
    if args.train_from_scratch:
        learning_rate = args.lr if args.lr != 5e-4 else 1e-3  # Default to 1e-3 for from scratch
        print(f"📈 Learning rate set to: {learning_rate} (training from scratch)")
    else:
        learning_rate = args.lr  # Use provided LR (default 5e-4 for fine-tuning)
        print(f"📈 Learning rate set to: {learning_rate} (fine-tuning)")
    
    new_lightning_module = ScaledMambaLightningModule(new_model_args, learning_rate=learning_rate)
    new_lightning_module.model = new_model
    
    # Setup training
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'mamba-{args.output_name}-{{epoch:02d}}-{{train_loss:.2f}}',
        save_top_k=3,
        monitor='train_loss'
    )
    
    # Configure trainer with TPU-safe settings
    def check_tpu_availability():
        """Check if TPU is actually available and accessible, not just importable."""
        try:
            import torch_xla
            print("✅ torch_xla imported successfully")
            
            try:
                import torch_xla.core.xla_model as xm
                print("✅ torch_xla.core.xla_model imported successfully")
                
                # Try to get TPU devices - but be more lenient
                try:
                    devices = xm.get_xla_supported_devices()
                    print(f"🔍 XLA supported devices: {devices}")
                    
                    if devices:
                        tpu_devices = [d for d in devices if 'TPU' in str(d)]
                        print(f"🔍 TPU devices found: {tpu_devices}")
                        if tpu_devices:
                            return True
                        else:
                            print("⚠️  No TPU devices in XLA supported devices")
                            # In Colab, sometimes TPU is available but not showing in devices
                            # Let's be more lenient and check environment
                            if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
                                print("🔍 TPU environment variables detected, assuming TPU is available")
                                return True
                            return False
                    else:
                        print("⚠️  No XLA devices found")
                        # Still check environment variables
                        if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
                            print("🔍 TPU environment variables detected, assuming TPU is available")
                            return True
                        return False
                        
                except Exception as device_error:
                    print(f"⚠️  Error getting XLA devices: {device_error}")
                    # Fallback to environment check
                    if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
                        print("🔍 TPU environment variables detected, assuming TPU is available")
                        return True
                    return False
                    
            except Exception as xm_error:
                print(f"⚠️  Error importing xla_model: {xm_error}")
                return False
                
        except Exception as e:
            print(f"⚠️  TPU check failed: {e}")
            return False

    is_tpu = check_tpu_availability()

    # Override TPU detection if force flag is set
    if args.force_tpu:
        print("🔧 Force TPU flag set - bypassing TPU detection")
        is_tpu = True

    trainer_args = {
        'max_epochs': args.epochs,
        'callbacks': [checkpoint_callback],
        'default_root_dir': 'logs_scaled',
        'enable_progress_bar': True,
        'log_every_n_steps': 50,
    }

    if is_tpu:
        print("🔥 Using TPU acceleration")
        trainer_args.update({
            'accelerator': 'tpu',
            'devices': 'auto',
            'precision': 32,
        })
    else:
        if torch.cuda.is_available():
            print("🚀 Using GPU acceleration")
            trainer_args.update({
                'accelerator': 'gpu',
                'devices': 'auto',
                'precision': '16-mixed',
            })
        else:
            print("💻 Using CPU (no accelerator available)")
            trainer_args.update({
                'accelerator': 'cpu',
                'devices': 'auto',
                'precision': 32,
            })

    trainer = pl.Trainer(**trainer_args)

    # Start training with fallback mechanism
    if args.train_from_scratch:
        print(f"\n🚀 Starting training from scratch...")
        print(f"   Model: {new_params/1e6:.1f}M params ({args.d_model}D, {args.n_layer}L)")
        print(f"   Training epochs: {args.epochs}")
        print(f"   Drop first column: {args.drop_first_column}")
    else:
        print(f"\n🚀 Starting scaled training...")
        print(f"   Source: {source_params/1e6:.1f}M params -> Target: {new_params/1e6:.1f}M params")
        print(f"   Additional epochs: {args.epochs}")
        print(f"   Drop first column: {args.drop_first_column}")

    try:
        trainer.fit(new_lightning_module, data_module)
    except Exception as e:
        if "TPU" in str(e) or "xla" in str(e).lower() or "accel" in str(e):
            print(f"\n⚠️  TPU training failed: {e}")
            print("🔄 Falling back to GPU/CPU training...")
            
            # Create new trainer with CPU/GPU fallback
            fallback_trainer_args = {
                'max_epochs': args.epochs,
                'callbacks': [checkpoint_callback],
                'default_root_dir': 'logs_scaled',
                'enable_progress_bar': True,
                'log_every_n_steps': 50,
            }
            
            if torch.cuda.is_available():
                print("🚀 Using GPU acceleration (fallback)")
                fallback_trainer_args.update({
                    'accelerator': 'gpu',
                    'devices': 'auto',
                    'precision': '16-mixed',
                })
            else:
                print("💻 Using CPU (fallback)")
                fallback_trainer_args.update({
                    'accelerator': 'cpu',
                    'devices': 'auto',
                    'precision': 32,
                })
            
            trainer = pl.Trainer(**fallback_trainer_args)
            trainer.fit(new_lightning_module, data_module)
        else:
            # Re-raise non-TPU related errors
            raise e
    
    # Save final model
    final_checkpoint = f"mamba_{args.output_name}_final.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"✅ Scaled model saved as: {final_checkpoint}")
    
    # Save tokenizer with new name
    tokenizer_file = f'tokenizer_{args.output_name}.pkl'
    decode_file = f'decode_{args.output_name}.pkl'
    
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(data_module.tokenizer, f)
    with open(decode_file, 'wb') as f:
        pickle.dump(data_module.decode, f)
    
    print(f"✅ Tokenizer saved as: {tokenizer_file}")
    print(f"✅ Decoder saved as: {decode_file}")
    
    # Print summary
    print(f"\n📊 Training Completed!")
    if args.train_from_scratch:
        print(f"   Model:     {new_params/1e6:.1f}M params ({args.d_model}D, {args.n_layer}L)")
        print(f"   Training:  From scratch ({args.epochs} epochs)")
        print(f"   Layers:    {len(skipped)} randomly initialized")
    else:
        print(f"   Original:  {source_params/1e6:.1f}M params")
        print(f"   Scaled:    {new_params/1e6:.1f}M params ({new_params/source_params:.1f}x)")
        print(f"   Transferred layers: {len(transferred)}")
        print(f"   New/expanded layers: {len(skipped)}")
    print(f"   Vocabulary: {data_module.vocab_size}")

if __name__ == '__main__':
    main() 