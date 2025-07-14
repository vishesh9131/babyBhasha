import torch
import pickle
from mamba_tiny_master.model import Mamba, ModelArgs
from colab.train import MambaLightningModule

def generate(prompt, model, tokenizer, decode, max_len=100):
    model.eval()
    input_ids = torch.tensor([tokenizer[c] for c in prompt]).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            # Get the logits for the last token
            next_token_logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.get('\n'):
                break

    return "".join(decode[id.item()] for id in input_ids[0])


def main():
    # Load the model from checkpoint
    checkpoint_path = "mamba_final.ckpt"
    model = MambaLightningModule.load_from_checkpoint(checkpoint_path).model

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('decode.pkl', 'rb') as f:
        decode = pickle.load(f)

    prompt = "The quick brown fox"
    generated_text = generate(prompt, model, tokenizer, decode)
    print(generated_text)

if __name__ == '__main__':
    main() 