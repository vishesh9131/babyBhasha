#!/usr/bin/env python3
"""
Quick Chat - Simple conversational interface with your trained Mamba model
Usage: python quick_chat.py [checkpoint_path]
"""

import torch
import pickle
import sys
import glob
import argparse
import os

sys.path.append('.')
from train import MambaLightningModule

def find_latest_checkpoint():
    """Find the most recent checkpoint file"""
    checkpoint_files = glob.glob('checkpoints/*.ckpt')
    if checkpoint_files:
        return max(checkpoint_files, key=lambda x: x)
    elif os.path.exists('mamba_final.ckpt'):
        return 'mamba_final.ckpt'
    else:
        return None

def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer from files"""
    print(f"Loading model from: {checkpoint_path}")
    model = MambaLightningModule.load_from_checkpoint(checkpoint_path).model
    model.eval()
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('decode.pkl', 'rb') as f:
        decode = pickle.load(f)
    
    return model, tokenizer, decode

def generate_text(model, tokenizer, decode, prompt, max_length=100, temperature=0.8):
    """Generate text from the model"""
    # Encode prompt
    input_ids = torch.tensor([tokenizer[c] for c in prompt if c in tokenizer]).unsqueeze(0)
    
    if input_ids.shape[1] == 0:
        return "Sorry, I can't understand that input."
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_logits = logits[0, -1, :] / temperature
            
            # Simple sampling
            probs = torch.softmax(next_logits, dim=0)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop at sentence end
            next_char = decode.get(next_token.item(), '')
            if next_char in ['\n', '.', '!', '?'] and generated.shape[1] > input_ids.shape[1] + 5:
                break
    
    # Decode response (only new part)
    response_tokens = generated[0, input_ids.shape[1]:]
    response = ''.join([decode.get(t.item(), '') for t in response_tokens])
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Quick chat with Mamba model")
    parser.add_argument('checkpoint', nargs='?', help="Path to checkpoint file")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature for generation")
    parser.add_argument('--length', type=int, default=100, help="Max response length")
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("❌ No checkpoint found! Train a model first.")
            return
    
    # Load model
    try:
        model, tokenizer, decode = load_model_and_tokenizer(checkpoint_path)
        print(f"✅ Model loaded! Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Chat loop
    print("\n" + "="*50)
    print("🎭 QUICK MAMBA CHAT")
    print("Type 'quit' to exit, 'help' for commands")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("Commands: quit, help")
                print(f"Settings: temp={args.temp}, max_length={args.length}")
                continue
            
            if not user_input:
                continue
            
            # Generate response
            print("🤖 Bot: ", end="", flush=True)
            response = generate_text(model, tokenizer, decode, user_input, 
                                   args.length, args.temp)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 