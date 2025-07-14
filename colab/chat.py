import torch
import pickle
import sys
import os
from typing import Optional

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
from train import MambaLightningModule

class MambaChat:
    def __init__(self, checkpoint_path: str = "mamba_final.ckpt", 
                 tokenizer_path: str = "tokenizer.pkl", 
                 decode_path: str = "decode.pkl"):
        """
        Initialize the Mamba chat interface.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer pickle file
            decode_path: Path to the decode mapping pickle file
        """
        print("🤖 Loading Mamba Chat Interface...")
        
        # Load the model
        try:
            print(f"Loading model from: {checkpoint_path}")
            self.lightning_module = MambaLightningModule.load_from_checkpoint(checkpoint_path)
            self.model = self.lightning_module.model
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load tokenizer and decoder
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(decode_path, 'rb') as f:
                self.decode = pickle.load(f)
            print(f"✅ Tokenizer loaded! Vocabulary size: {len(self.tokenizer)}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
        
        # Chat settings
        self.max_response_length = 200
        self.temperature = 0.8
        self.top_k = 40
        self.context_length = 512
        
        # Conversation history
        self.conversation_history = ""
        
        print("🎉 Mamba Chat is ready!")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to token IDs"""
        # Handle unknown characters by skipping them
        token_ids = []
        for char in text:
            if char in self.tokenizer:
                token_ids.append(self.tokenizer[char])
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text"""
        if len(token_ids.shape) > 1:
            token_ids = token_ids.squeeze(0)
        
        text = ""
        for token_id in token_ids:
            token_id = token_id.item()
            if token_id in self.decode:
                text += self.decode[token_id]
        return text
    
    def generate_response(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated response
            
        Returns:
            Generated text response
        """
        if max_length is None:
            max_length = self.max_response_length
        
        # Encode the prompt
        input_ids = self.encode_text(prompt)
        
        # Truncate context if too long
        if input_ids.shape[1] > self.context_length:
            input_ids = input_ids[:, -self.context_length:]
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.model(generated_ids)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / self.temperature
                
                # Apply top-k filtering
                if self.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, self.top_k)
                    # Set all other logits to very negative values
                    next_token_logits.fill_(-float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample from the distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for stopping conditions
                next_char = self.decode.get(next_token.item(), '')
                if next_char in ['\n', '.', '!', '?'] and len(generated_ids[0]) > input_ids.shape[1] + 10:
                    break
                
                # Prevent infinite loops
                if generated_ids.shape[1] > input_ids.shape[1] + max_length:
                    break
        
        # Decode the response (only the new part)
        response_ids = generated_ids[:, input_ids.shape[1]:]
        response = self.decode_tokens(response_ids)
        
        return response.strip()
    
    def update_conversation_history(self, user_input: str, bot_response: str):
        """Update the conversation history"""
        self.conversation_history += f"Human: {user_input}\nAssistant: {bot_response}\n"
        
        # Keep only recent conversation (last 1000 characters)
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-1000:]
    
    def chat_loop(self):
        """Main chat loop"""
        print("\n" + "="*60)
        print("🎭 MAMBA CHAT - Interactive Conversation")
        print("="*60)
        print("💡 Tips:")
        print("  - Type 'quit', 'exit', or 'bye' to end the chat")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'settings' to adjust generation parameters")
        print("  - Keep inputs relatively short for better responses")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = ""
                    print("🧹 Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'settings':
                    self.adjust_settings()
                    continue
                
                elif not user_input:
                    print("Please enter something to continue the conversation.")
                    continue
                
                # Create context-aware prompt
                if self.conversation_history:
                    # Use recent conversation as context
                    prompt = self.conversation_history + f"Human: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(prompt)
                
                if response:
                    print(response)
                    # Update conversation history
                    self.update_conversation_history(user_input, response)
                else:
                    print("I'm having trouble generating a response. Try rephrasing your message.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")
                print("Please try again with a different input.")
    
    def adjust_settings(self):
        """Allow user to adjust generation settings"""
        print("\n⚙️ Current Settings:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Max response length: {self.max_response_length}")
        
        try:
            temp = input(f"\nNew temperature (0.1-2.0, current: {self.temperature}): ").strip()
            if temp:
                self.temperature = max(0.1, min(2.0, float(temp)))
            
            top_k = input(f"New top-k (1-100, current: {self.top_k}): ").strip()
            if top_k:
                self.top_k = max(1, min(100, int(top_k)))
            
            max_len = input(f"New max response length (10-500, current: {self.max_response_length}): ").strip()
            if max_len:
                self.max_response_length = max(10, min(500, int(max_len)))
            
            print("✅ Settings updated!")
            
        except ValueError:
            print("❌ Invalid input. Settings unchanged.")

def main():
    """Main function to start the chat interface"""
    print("🚀 Starting Mamba Chat Interface...")
    
    # Check if required files exist
    required_files = ["mamba_final.ckpt", "tokenizer.pkl", "decode.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you have trained a model first using train.py")
        return
    
    try:
        # Initialize chat interface
        chat = MambaChat()
        
        # Start chat loop
        chat.chat_loop()
        
    except Exception as e:
        print(f"❌ Failed to start chat interface: {e}")
        print("Make sure you have trained a model and the checkpoint files exist.")

if __name__ == "__main__":
    main() 