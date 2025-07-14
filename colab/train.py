import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import requests

import sys
import os

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

class TinyShakespeareDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, seq_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = None
        self.vocab_size = None
        self.dataset = None

    def prepare_data(self):
        # this is a dummy function i will use in future to inject different datasets onto running ckpt
        # for now it is just a placeholder
        # vishesh
        pass

    def setup(self, stage=None):
        # Download Tiny Shakespeare dataset
        try:
            print("Downloading Tiny Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
            print(f"Downloaded {len(text)} characters of Shakespeare text")
        except Exception as e:
            print(f"Failed to download Shakespeare dataset: {e}")
            print("Using fallback sample text for demonstration...")
            # Fallback to sample text
            text = """
            To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune,
            Or to take arms against a sea of troubles
            And by opposing end them. To die—to sleep,
            No more; and by a sleep to say we end
            That flesh is heir to: 'tis a consummation
            Devoutly to be wish'd. To die, to sleep;
            To sleep, perchance to dream—ay, there's the rub:
            For in that sleep of death what dreams may come,
            When we have shuffled off this mortal coil,
            Must give us pause—there's the respect
            That makes calamity of so long life.
            """ * 100  # Repeat to get more training data
        
        # Build vocabulary from characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create character-level tokenizer
        self.tokenizer = {ch: i for i, ch in enumerate(chars)}
        self.decode = {i: ch for i, ch in enumerate(chars)}

        # Tokenize the entire text
        tokenized_ids = [self.tokenizer[c] for c in text]

        # Create training examples by sliding window
        examples = []
        for i in range(0, len(tokenized_ids) - self.seq_len, self.seq_len):
            examples.append(torch.tensor(tokenized_ids[i:i + self.seq_len], dtype=torch.long))
            
        self.dataset = examples
        print(f"Created {len(examples)} training examples")

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

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
    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Model arguments for a ~1M parameter model
    data_module = TinyShakespeareDataModule(batch_size=16, seq_len=128)
    # We need to setup the data module to get vocab_size
    data_module.setup()

    model_args = ModelArgs(
        d_model=512,
        n_layer=8,
        vocab_size=data_module.vocab_size, 
        expand=2,
    )
    
    # Check if a checkpoint exists
    import glob
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='mamba-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        monitor='train_loss'
    )

    model = MambaLightningModule(model_args)

    # Look for existing checkpoints
    checkpoint_files = glob.glob('checkpoints/*.ckpt')
    resume_from_checkpoint = None
    
    if checkpoint_files:
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Found existing checkpoint: {latest_checkpoint}")
        resume_from_checkpoint = latest_checkpoint
    else:
        print("No existing checkpoints found, starting fresh training")

    # Configure trainer with GPU support
    trainer = pl.Trainer(
        max_epochs=15,  # Increased from 5 to 15 for continued training
        callbacks=[checkpoint_callback],
        default_root_dir='logs',
        accelerator='auto',  # This should auto-detect GPU
        devices='auto',      # This should use available devices
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Use mixed precision on GPU
    )
    
    # Start training (will resume if checkpoint is found)
    trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)
    
    # Save the final model
    trainer.save_checkpoint("mamba_final.ckpt")
    
    # Save the tokenizer
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(data_module.tokenizer, f)
    with open('decode.pkl', 'wb') as f:
        pickle.dump(data_module.decode, f)


if __name__ == '__main__':
    main() 