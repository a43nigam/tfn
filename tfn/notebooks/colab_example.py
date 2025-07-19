"""
Google Colab Example for TFN Layer

This example shows how to use the TFN layer in Google Colab.
Copy the tfn_layer.py content and this example to get started quickly.

Usage in Colab:
1. Copy tfn_layer.py content to a cell and run it
2. Copy this example to another cell and run it
3. The TFN layer is ready to use!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the TFN layer functions
from tfn_layer import tfn_layer

def simple_tfn_example():
    """Simple example showing TFN layer usage."""
    print("🚀 TFN Layer Colab Example")
    print("=" * 40)
    
    # Create sample data (like token embeddings from a transformer)
    batch_size, seq_len, embed_dim = 2, 10, 128
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.linspace(0.1, 0.9, seq_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
    
    print(f"Input embeddings shape: {embeddings.shape}")
    print(f"Token positions shape: {positions.shape}")
    
    # Apply TFN layer with different configurations
    print("\n📊 Testing different TFN configurations:")
    
    configs = [
        ("rbf", "cnn", "RBF kernel + CNN evolution"),
        ("compact", "spectral", "Compact kernel + Spectral evolution"),
        ("fourier", "pde", "Fourier kernel + PDE evolution")
    ]
    
    for kernel_type, evolution_type, description in configs:
        print(f"\n🔧 {description}")
        
        # Apply TFN layer
        updated_embeddings = tfn_layer(
            embeddings, positions,
            kernel_type=kernel_type,
            evolution_type=evolution_type,
            grid_size=50,  # Smaller grid for faster demo
            time_steps=2
        )
        
        # Check shape preservation
        assert updated_embeddings.shape == embeddings.shape
        print(f"✅ Shape preserved: {updated_embeddings.shape}")
        
        # Check that the layer actually changes the embeddings
        change_norm = torch.norm(updated_embeddings - embeddings).item()
        print(f"📈 Embedding change: {change_norm:.4f}")
        
        # Test gradient flow
        embeddings.requires_grad_(True)
        updated = tfn_layer(embeddings, positions, kernel_type, evolution_type)
        loss = updated.sum()
        loss.backward()
        grad_norm = torch.norm(embeddings.grad).item()
        print(f"🔗 Gradient flow: {grad_norm:.4f}")


def tfn_in_transformer_style():
    """Show how TFN can be used as a drop-in replacement for attention."""
    print("\n🔄 TFN as Attention Replacement")
    print("=" * 40)
    
    # Simulate a transformer layer
    batch_size, seq_len, embed_dim = 1, 8, 64
    
    # Input embeddings (like from a transformer)
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.linspace(0.1, 0.9, seq_len).unsqueeze(0).unsqueeze(-1)
    
    print(f"Original embeddings: {x.shape}")
    print(f"Sample values: {x[0, :3, :3]}")
    
    # Apply TFN layer (replaces self-attention)
    x_tfn = tfn_layer(x, positions, kernel_type="rbf", evolution_type="cnn")
    
    print(f"After TFN: {x_tfn.shape}")
    print(f"Sample values: {x_tfn[0, :3, :3]}")
    
    # Show the transformation effect
    change = torch.norm(x_tfn - x, dim=-1)
    print(f"Change per token: {change[0]}")
    
    return x_tfn


def multi_sentence_example():
    """Show TFN handling multiple sentences with different lengths."""
    print("\n📝 Multi-Sentence Processing")
    print("=" * 40)
    
    # Create sentences of different lengths
    sentences = [
        "Hello world",  # 2 tokens
        "The quick brown fox jumps",  # 5 tokens
        "A very long sentence with many tokens to process"  # 9 tokens
    ]
    
    # Simulate tokenized embeddings (pad to max length)
    max_len = 9
    batch_size = len(sentences)
    embed_dim = 64
    
    # Create embeddings (simulating tokenizer output)
    embeddings = torch.randn(batch_size, max_len, embed_dim)
    positions = torch.linspace(0.1, 0.9, max_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
    
    print(f"Processing {batch_size} sentences, max length: {max_len}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Apply TFN layer
    updated = tfn_layer(embeddings, positions, kernel_type="compact", evolution_type="spectral")
    
    print(f"✅ Updated shape: {updated.shape}")
    
    # Show that each sentence is processed independently
    for i, sentence in enumerate(sentences):
        tokens = len(sentence.split())
        change = torch.norm(updated[i, :tokens] - embeddings[i, :tokens], dim=-1).mean()
        print(f"'{sentence}': {tokens} tokens, avg change: {change:.4f}")


if __name__ == "__main__":
    # Run examples
    simple_tfn_example()
    tfn_in_transformer_style()
    multi_sentence_example()
    
    print("\n🎉 TFN Layer is ready for Colab deployment!")
    print("\nTo use in Colab:")
    print("1. Copy tfn_layer.py content to a cell")
    print("2. Copy this example to another cell")
    print("3. Run both cells")
    print("4. Use tfn_layer() function directly!") 