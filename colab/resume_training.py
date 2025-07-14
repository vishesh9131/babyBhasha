import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import requests
import glob
import os

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
from train import TinyShakespeareDataModule, MambaLightningModule

def resume_training(checkpoint_path=None, additional_epochs=10):
    """Resume training from a checkpoint"""
    
    # Set up data module
    data_module = TinyShakespeareDataModule(batch_size=16, seq_len=128)
    data_module.setup()
    
    # Find checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_files = glob.glob('checkpoints/*.ckpt')
        if not checkpoint_files:
            print("No checkpoints found! Please train a model first.")
            return
        checkpoint_path = max(checkpoint_files, key=os.path.getctime)
    
    print(f"Resuming training from: {checkpoint_path}")
    
    # Load the model from checkpoint
    model = MambaLightningModule.load_from_checkpoint(checkpoint_path)
    
    # Extract current epoch from checkpoint path if possible
    try:
        import re
        epoch_match = re.search(r'epoch=(\d+)', checkpoint_path)
        current_epoch = int(epoch_match.group(1)) if epoch_match else 0
    except:
        current_epoch = 0
    
    max_epochs = current_epoch + additional_epochs
    print(f"Current epoch: {current_epoch}, will train until epoch: {max_epochs}")
    
    # Set up checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='mamba-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,  # Keep top 3 checkpoints
        monitor='train_loss'
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir='logs',
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
    )
    
    # Resume training
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    
    # Save final model
    trainer.save_checkpoint("mamba_final.ckpt")
    print("Training completed!")

if __name__ == '__main__':
    # You can specify a specific checkpoint path or let it find the latest one
    # resume_training(checkpoint_path='checkpoints/mamba-epoch=04-train_loss=X.XX.ckpt', additional_epochs=10)
    resume_training(additional_epochs=10) 