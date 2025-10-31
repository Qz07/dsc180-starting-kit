import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
from tqdm import tqdm

def get_model_layers(model):
    """
    Recursively flattens a model into a list of (name, layer) tuples,
    filtering for specific types of layers that are suitable for
    sampling activations.
    
    This works for any model architecture, not just ResNet.
    """
    
    # --- Customize this list ---
    # Define the types of layers you want to be able to sample from.
    # These are typically the layers that produce the main representations.
    SAMPLABLE_LAYER_TYPES = (
        nn.Conv2d,
        nn.Linear
    )
    
    samplable_layers = []
    
    # model.named_modules() recursively finds EVERY module in the network
    # (e.g., 'layer1.0.conv1', 'layer1.0.bn1', etc.)
    for name, module in model.named_modules():
        # We check if the module is an instance of any of the types
        # we defined in our tuple above.
        if isinstance(module, SAMPLABLE_LAYER_TYPES):
            samplable_layers.append((name, module))
            
    return samplable_layers

# def get_model_layers(model):
#     """
#     Flattens a sequential or ResNet-like model into a
#     list of (name, layer) tuples.
#     """
#     layers = []
#     # Example for a simple Sequential model
#     if isinstance(model, nn.Sequential):
#         for i, layer in enumerate(model.children()):
#             layers.append((f"layer_{i}", layer))
#     # Example for ResNet
#     elif hasattr(model, 'conv1'):
#         layers.append(("conv1", model.conv1))
#         layers.append(("bn1", model.bn1))
#         layers.append(("relu", model.relu))
#         layers.append(("maxpool", model.maxpool))
        
#         if hasattr(model, 'layer1'):
#             layers.append(("layer1", model.layer1))
#         if hasattr(model, 'layer2'):
#             layers.append(("layer2", model.layer2))
#         if hasattr(model, 'layer3'):
#             layers.append(("layer3", model.layer3))
#         if hasattr(model, 'layer4'):
#             layers.append(("layer4", model.layer4))
            
#         layers.append(("avgpool", model.avgpool))
#         layers.append(("fc", model.fc))
#     else:
#         # Fallback for other models
#         for name, layer in model.named_children():
#             layers.append((name, layer))
            
#     # We only want to sample from layers that produce activations
#     # (e.g., Conv, Linear, or blocks) not non-linearities or pooling.
#     # For this example, let's keep it simple and use all named children.
#     # In a real case, you would filter this list.
    
#     return [(name, layer) for name, layer in model.named_children() if 'conv' in name or 'layer' in name or 'fc' in name]


def get_layer_range_indices(k, num_layers):
    """
    Returns the (start, end) indices for layer sampling based on k.
    """
    if k == 0.25:
        # First 25%
        start_idx = 0
        end_idx = int(num_layers * 0.25)-1
    elif k == 0.50:
        # 20% to 50% (as you specified)
        start_idx = int(num_layers * 0.25)
        end_idx = int(num_layers * 0.50)-1
    elif k == 0.75:
        # ASSUMPTION: 50% to 75%
        # print("Assuming k=0.75 samples from 50%-75% of layers.")
        start_idx = int(num_layers * 0.50)
        end_idx = int(num_layers * 0.75)-1
    elif k == 1.0:
        # ASSUMPTION: 75% to 100%
        # print("Assuming k=1.0 samples from 75%-100% of layers.")
        start_idx = int(num_layers * 0.75)
        end_idx = num_layers - 1 # Use -1 for zero-based index
    else:
        raise ValueError(f"Unknown k value: {k}")

    # Ensure end_idx is at least start_idx and within bounds
    end_idx = max(start_idx + 1, end_idx) # Ensure at least one layer
    end_idx = min(end_idx, num_layers - 1)
    
    return start_idx, end_idx

def create_random_vectors(model_layers, device='cpu'):
    """
    Creates a dictionary of random unit vectors, one for each
    layer's activation. This is needed because each layer may
    have a different number of output channels.
    """
    u_vectors = {}
    for name, layer in model_layers:
        try:
            # Get output channels for Conv/Linear layers
            if hasattr(layer, 'out_channels'):
                channels = layer.out_channels
            elif hasattr(layer, 'out_features'):
                channels = layer.out_features
            # Handle ResNet blocks (use the output channels of the last conv)
            elif isinstance(layer, nn.Sequential):
                last_conv = [m for m in layer.modules() if isinstance(m, (nn.Conv2d, nn.Linear))][-1]
                channels = last_conv.out_channels if hasattr(last_conv, 'out_channels') else last_conv.out_features
            else:
                # Skip layers we can't get channels for
                continue

            # Create a random vector u
            u = torch.rand(channels, device=device)
            # Normalize to make it a unit vector
            u = u / torch.norm(u)
            
            # Reshape for broadcasting with (B, C, H, W) activations
            # New shape: (1, C, 1, 1)
            u_vectors[name] = u.view(1, -1, 1, 1)
            
        except Exception as e:
            print(f"Skipping layer {name}: {e}")
            
    return u_vectors

def train_rmu(model, forget_loader, retain_loader, k, epochs=1, lr=1e-3, c=6.5, alpha=1200, device='cpu', seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.to(device)
    model.train()
    
    # 1. Create the frozen model
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    for param in model_frozen.parameters():
        param.requires_grad = False
        
    # 2. Setup optimizer (only for the updated model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Get the list of layers to sample from
    model_layers = get_model_layers(model) 
    layer_names = [name for name, _ in model_layers]
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("Error: get_model_layers() returned 0 samplable layers. Check your SAMPLABLE_LAYER_TYPES.")
        return model # Return the original model

    # 4. Create the dictionary of random 'u' vectors
    u_vectors = create_random_vectors(model_layers, device=device)
    
    # Store activations from the hook
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    # --- Training Loop ---
    for epoch in tqdm(range(epochs)):
        
        # --- 5. Sample ONE layer for the ENTIRE epoch ---
        start_idx, end_idx = get_layer_range_indices(k, num_layers)
        
        if start_idx >= end_idx or start_idx >= num_layers or end_idx > num_layers:
            print(f"Warning: Invalid layer range ({start_idx}, {end_idx}) for k={k} with {num_layers} layers. Skipping epoch {epoch+1}.")
            continue
            
        sampled_idx = random.randint(start_idx, end_idx)
        sampled_layer_name = layer_names[sampled_idx]
        
        if sampled_layer_name not in u_vectors:
            print(f"Warning: Layer {sampled_layer_name} has no 'u' vector. Skipping epoch {epoch+1}.")
            continue

        # print(f"--- Starting Epoch {epoch+1}/{epochs}, Training on Layer: {sampled_layer_name} ---")

        # 6. Get the 'u' vector for the sampled layer
        u = u_vectors[sampled_layer_name].to(device)
        
        # 7. Register hooks ONCE for the epoch
        hook_handle_updated = None
        hook_handle_frozen = None
        try:
            hook_handle_updated = model.get_submodule(sampled_layer_name).register_forward_hook(get_activation("updated"))
            hook_handle_frozen = model_frozen.get_submodule(sampled_layer_name).register_forward_hook(get_activation("frozen"))
        except AttributeError:
            print(f"Error: Could not find submodule '{sampled_layer_name}'. Skipping epoch {epoch+1}.")
            if hook_handle_updated: hook_handle_updated.remove() # Clean up just in case
            continue
            
        # --- Batch Loop ---
        # Iterate over all batches using the *same* layer hook
        for (forget_data, _) , (retain_data, _) in zip(forget_loader, retain_loader):
            forget_data = forget_data.to(device)
            retain_data = retain_data.to(device)
            
            optimizer.zero_grad()
            
            # We MUST clear activations dict here so old batch values are gone
            activations = {} 

            # --- 8. Calculate Forget Loss ---
            _ = model(forget_data)
            act_forget = activations["updated"] # Get activation from hook
            
            # Handle FC layer (B, C) vs Conv layer (B, C, H, W)
            u_reshaped = u # Reset u_reshaped for each batch (in case of dim mismatch)
            if act_forget.dim() == 2: # (Batch, Features)
                u_reshaped = u.view(1, -1) # Reshape u to (1, Features)

            loss_forget = torch.mean((act_forget - c * u_reshaped)**2)

            # --- 9. Calculate Retain Loss ---
            # Run frozen model first to populate 'frozen' activation
            _ = model_frozen(retain_data)
            act_retain_frozen = activations["frozen"]
            
            # Run updated model to populate 'updated' activation
            _ = model(retain_data)
            act_retain_updated = activations["updated"] # Overwrites forget activation
            
            loss_retain = torch.mean((act_retain_updated - act_retain_frozen)**2)
            
            # --- 10. Total Loss and Backpropagation ---
            loss = loss_forget + alpha * loss_retain
            
            loss.backward()
            optimizer.step()
            
            # (No hook removal here)
            
        # --- 11. Remove hooks AFTER all batches for the epoch are done ---
        if hook_handle_updated: hook_handle_updated.remove()
        if hook_handle_frozen: hook_handle_frozen.remove()
        
        # print(f"--- Finished Epoch {epoch+1}/{epochs} ---")


    print("--- RMU Training Complete ---")
    
    # Return the model that has been unlearned
    return model
