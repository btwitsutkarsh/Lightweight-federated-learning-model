import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from models.net import Net

def compress_trained_model(model_path, device, test_dataset, batch_size=32, compression_rate=0.3):
    """Compress model using weight pruning and floating-point compression"""
    print("\nLoading trained model for compression...")
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Load model
    checkpoint = torch.load(model_path, weights_only=True)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get initial metrics
    initial_size = get_model_size(model)
    initial_accuracy = test_model(model, test_loader, device)
    
    print(f"\nInitial model size: {initial_size:.2f} MB")
    print(f"Initial accuracy: {initial_accuracy:.2f}%")
    
    # 1. Magnitude-based Weight Pruning
    print("\nApplying weight pruning...")
    pruned_state_dict = OrderedDict()
    
    for name, param in model.state_dict().items():
        if 'weight' in name:
            # Calculate threshold for pruning
            abs_weights = torch.abs(param)
            threshold = torch.quantile(abs_weights, compression_rate)
            
            # Create mask for significant weights
            mask = abs_weights > threshold
            
            # Store only significant weights and their indices
            values = param[mask]
            indices = torch.nonzero(mask)
            
            pruned_state_dict[name] = {
                'values': values.half(),  # Store as float16
                'indices': indices,
                'shape': param.shape,
                'mask': mask  # Store the mask for reconstruction
            }
        else:
            # For biases, just convert to float16
            pruned_state_dict[name] = param.half()
    
    # 2. Apply compression
    compressed_size = 0
    for name, tensor in pruned_state_dict.items():
        if isinstance(tensor, dict):
            compressed_size += (tensor['values'].nelement() * tensor['values'].element_size() +
                              tensor['indices'].nelement() * tensor['indices'].element_size())
        else:
            compressed_size += tensor.nelement() * tensor.element_size()
    
    compressed_size = compressed_size / (1024 * 1024)  # Convert to MB
    
    # 3. Reconstruct model for testing
    reconstructed_state_dict = OrderedDict()
    for name, tensor in pruned_state_dict.items():
        if isinstance(tensor, dict):
            # Reconstruct sparse tensor using the mask
            reconstructed = torch.zeros(tensor['shape'], dtype=torch.float32, device=device)
            reconstructed[tensor['mask']] = tensor['values'].float()
            reconstructed_state_dict[name] = reconstructed
        else:
            reconstructed_state_dict[name] = tensor.float()
    
    model.load_state_dict(reconstructed_state_dict)
    final_accuracy = test_model(model, test_loader, device)
    
    # Calculate sparsity
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0
    
    # Save compressed model
    compressed_path = model_path.replace('.pth', '_compressed.pth')
    torch.save({
        'pruned_state_dict': pruned_state_dict,
        'compression_metrics': {
            'original_size': initial_size,
            'compressed_size': compressed_size,
            'compression_ratio': (initial_size - compressed_size) / initial_size * 100,
            'sparsity': sparsity,
            'original_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy
        }
    }, compressed_path)
    
    print("\nCompression Results:")
    print(f"Original size: {initial_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {((initial_size - compressed_size) / initial_size * 100):.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Original accuracy: {initial_accuracy:.2f}%")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print(f"Accuracy change: {final_accuracy - initial_accuracy:+.2f}%")
    
    return model

def get_model_size(model):
    """Calculate model size in MB"""
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    return size / (1024 * 1024)

def test_model(model, test_loader, device):
    """Test model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return 100. * correct / total