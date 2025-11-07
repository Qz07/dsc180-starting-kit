#!/usr/bin/env python3
# rmu_train.py
import argparse
import copy
import random
from itertools import cycle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# --------------------------
# Layer utilities
# --------------------------
def get_model_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Recursively returns [(name, module)] for layers that produce sampleable activations:
    Conv2d and Linear. Works across most CNN backbones.
    """
    SAMPLABLE_LAYER_TYPES = (nn.Conv2d, nn.Linear)
    samplable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, SAMPLABLE_LAYER_TYPES):
            samplable_layers.append((name, module))
    return samplable_layers


def get_layer_range_indices(k: float, num_layers: int) -> Tuple[int, int]:
    """
    Maps k ∈ {0.25, 0.50, 0.75, 1.0} to index ranges over [0, num_layers-1], inclusive.
    """
    if k == 0.25:
        start_idx = 0
        end_idx = int(num_layers * 0.25) - 1
    elif k == 0.50:
        start_idx = int(num_layers * 0.25)
        end_idx = int(num_layers * 0.50) - 1
    elif k == 0.75:
        start_idx = int(num_layers * 0.50)
        end_idx = int(num_layers * 0.75) - 1
    elif k == 1.0:
        start_idx = int(num_layers * 0.75)
        end_idx = num_layers - 1
    else:
        raise ValueError(f"Unknown k value: {k}")

    # Ensure valid & non-empty range
    start_idx = max(0, min(start_idx, num_layers - 1))
    end_idx = max(start_idx, min(end_idx, num_layers - 1))
    return start_idx, end_idx


def create_random_vectors(model_layers: List[Tuple[str, nn.Module]], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Creates a dict: layer_name -> random unit vector u with shape:
      Conv: (1, C, 1, 1)
      Linear: stored as (1, C, 1, 1) then reshaped on-the-fly to (1, C)
    """
    u_vectors: Dict[str, torch.Tensor] = {}
    for name, layer in model_layers:
        try:
            if hasattr(layer, "out_channels"):
                channels = layer.out_channels
            elif hasattr(layer, "out_features"):
                channels = layer.out_features
            else:
                continue
            u = torch.rand(channels, device=device)
            u = u / (torch.norm(u) + 1e-12)
            u_vectors[name] = u.view(1, -1, 1, 1)
        except Exception as e:
            print(f"[create_random_vectors] Skipping layer {name}: {e}")
    return u_vectors


# --------------------------
# Misc utils
# --------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pair_full_epoch(forget_loader, retain_loader):
    """
    Pairs batches so the longer loader fully participates by cycling the shorter one.
    """
    len_f, len_r = len(forget_loader), len(retain_loader)
    if len_f >= len_r:
        main_iter = zip(forget_loader, cycle(retain_loader))
        num_steps = len_f
    else:
        main_iter = zip(cycle(forget_loader), retain_loader)
        num_steps = len_r
    return main_iter, num_steps


def parse_k_schedule(k_schedule_str: str, epochs: int) -> List[float]:
    """
    Parses a comma-separated k schedule; if fewer than epochs, the last value repeats.
    """
    if not k_schedule_str:
        # reasonable default: later layers
        ks = [0.75] + [1.0] * max(epochs - 1, 0)
        return ks[:epochs]
    parts = [float(x.strip()) for x in k_schedule_str.split(",") if x.strip()]
    if not parts:
        parts = [1.0]
    if len(parts) < epochs:
        parts = parts + [parts[-1]] * (epochs - len(parts))
    return parts[:epochs]


def linear_alpha(epoch: int, epochs: int, alpha: float, alpha_start: float, alpha_end: float) -> float:
    """
    Returns alpha per-epoch: either constant alpha (if alpha_start<0) or linear ramp.
    """
    if alpha_start >= 0 and alpha_end >= 0:
        if epochs <= 1:
            return alpha_end
        t = epoch / (epochs - 1)
        return alpha_start + t * (alpha_end - alpha_start)
    return alpha


def compute_layer_rms(
    model_frozen: nn.Module,
    layer_name: str,
    data_loader,
    device: str,
    num_batches: int = 2,
) -> float:
    """
    Estimates RMS activation magnitude for the sampled layer using the frozen model
    across a few batches of *retain* data.
    """
    activations = {}

    def hook_fn(_m, _inp, out):
        activations["rms"] = out

    handle = model_frozen.get_submodule(layer_name).register_forward_hook(hook_fn)
    model_frozen.eval()
    cnt = 0
    with torch.no_grad():
        for (data, _) in data_loader:
            data = data.to(device)
            _ = model_frozen(data)
            if "rms" in activations:
                act = activations["rms"].detach()
                rms = torch.sqrt(torch.mean(act ** 2)).item()
                handle.remove()
                return rms
            cnt += 1
            if cnt >= num_batches:
                break
    handle.remove()
    # Fallback if no activation captured
    return 1.0


# --------------------------
# RMU training
# --------------------------
def train_rmu(
    model: nn.Module,
    forget_loader,
    retain_loader,
    epochs: int = 4,
    k_schedule: List[float] = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    alpha: float = 1000.0,
    alpha_start: float = -1.0,
    alpha_end: float = -1.0,
    c: float = 4.0,
    auto_scale_c: bool = False,
    c_scale: float = 3.0,
    grad_clip: float = 1.0,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> nn.Module:
    set_all_seeds(seed)
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model = model.to(device)
    model.train()

    # 1) Frozen teacher
    model_frozen = copy.deepcopy(model).to(device)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad = False

    # 2) Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3) Discover layers & u-vectors
    model_layers = get_model_layers(model)
    layer_names = [n for n, _ in model_layers]
    num_layers = len(layer_names)
    if num_layers == 0:
        raise RuntimeError("No samplable layers found. Check get_model_layers().")
    u_vectors = create_random_vectors(model_layers, device=device)

    # 4) Hooks will write here
    activations = {}

    def get_activation(name):
        def hook(_m, _inp, out):
            activations[name] = out
        return hook

    # 5) Per-epoch loop
    if k_schedule is None:
        k_schedule = [0.75] + [1.0] * max(epochs - 1, 0)

    for epoch in range(epochs):
        k = k_schedule[epoch] if epoch < len(k_schedule) else k_schedule[-1]
        # pick index range for this epoch
        start_idx, end_idx = get_layer_range_indices(k, num_layers)
        if start_idx > end_idx:
            if verbose:
                print(f"[Epoch {epoch+1}] Invalid range ({start_idx},{end_idx}) for k={k}. Skipping.")
            continue

        sampled_idx = random.randint(start_idx, end_idx)
        sampled_layer_name = layer_names[sampled_idx]
        if sampled_layer_name not in u_vectors:
            if verbose:
                print(f"[Epoch {epoch+1}] No u-vector for {sampled_layer_name}. Skipping.")
            continue

        # Alpha (retain strength) for this epoch
        alpha_e = linear_alpha(epoch, epochs, alpha, alpha_start, alpha_end)

        # Optionally auto-scale c using frozen activations RMS
        if auto_scale_c:
            rms = compute_layer_rms(model_frozen, sampled_layer_name, retain_loader, device)
            c_e = c_scale * max(rms, 1e-6)
        else:
            c_e = c

        # Register hooks
        hook_updated = model.get_submodule(sampled_layer_name).register_forward_hook(get_activation("updated"))
        hook_frozen = model_frozen.get_submodule(sampled_layer_name).register_forward_hook(get_activation("frozen"))

        # Pair loaders fully
        main_iter, num_steps = pair_full_epoch(forget_loader, retain_loader)

        if verbose:
            print(
                f"[Epoch {epoch+1}/{epochs}] layer='{sampled_layer_name}' "
                f"range=[{start_idx},{end_idx}] steps={num_steps} "
                f"alpha={alpha_e:.2f} c={c_e:.4f} k={k}"
            )

        # Batch loop
        pbar = tqdm(total=num_steps, leave=False)
        for (forget_batch, retain_batch) in main_iter:
            (forget_data, _f_y) = forget_batch
            (retain_data, _r_y) = retain_batch
            forget_data = forget_data.to(device)
            retain_data = retain_data.to(device)

            optimizer.zero_grad(set_to_none=True)
            activations.clear()

            # Forget loss: push updated activations toward c * u
            _ = model(forget_data)
            act_f = activations.get("updated", None)
            if act_f is None:
                # very unlikely; skip this step
                continue

            u = u_vectors[sampled_layer_name].to(device)
            if act_f.dim() == 2:  # (B, C)
                u_reshaped = u.view(1, -1)
            else:  # (B, C, H, W)
                u_reshaped = u
            loss_forget = torch.mean((act_f - c_e * u_reshaped) ** 2)

            # Retain loss: match frozen activations on retain data
            _ = model_frozen(retain_data)
            act_r_frozen = activations.get("frozen", None)
            _ = model(retain_data)
            act_r_updated = activations.get("updated", None)
            if act_r_frozen is None or act_r_updated is None:
                continue
            loss_retain = torch.mean((act_r_updated - act_r_frozen) ** 2)

            loss = loss_forget + alpha_e * loss_retain
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            pbar.update(1)
        pbar.close()

        # Remove hooks
        hook_updated.remove()
        hook_frozen.remove()

    if verbose:
        print("--- RMU Training Complete ---")
    return model


# --------------------------
# CLI
# --------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Run RMU on a model with CIFAR-10 (or similar) loaders.")
    # Core training
    p.add_argument("--epochs", type=int, default=4, help="Number of RMU epochs.")
    p.add_argument(
        "--k_schedule",
        type=str,
        default="0.75,1.0,1.0,1.0",
        help="Comma-separated k per epoch (values in {0.25,0.5,0.75,1.0}). "
             "If shorter than epochs, last value repeats.",
    )
    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam.")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay.")
    # Retain strength (alpha)
    p.add_argument("--alpha", type=float, default=1000.0, help="Constant alpha if alpha_start/end not set.")
    p.add_argument(
        "--alpha_start",
        type=float,
        default=-1.0,
        help="If >=0 with alpha_end>=0, use linear ramp from alpha_start to alpha_end.",
    )
    p.add_argument(
        "--alpha_end",
        type=float,
        default=-1.0,
        help="If >=0 with alpha_start>=0, use linear ramp from alpha_start to alpha_end.",
    )
    # Target magnitude c
    p.add_argument("--c", type=float, default=4.0, help="Target activation scale c (ignored if auto_scale_c).")
    p.add_argument("--auto_scale_c", action="store_true", help="Auto-scale c from frozen activation RMS.")
    p.add_argument(
        "--c_scale",
        type=float,
        default=3.0,
        help="Multiplier applied to RMS activation if --auto_scale_c is set.",
    )
    # Misc
    p.add_argument("--grad_clip", type=float, default=1.0, help="Grad clipping max-norm (<=0 disables).")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no_verbose", action="store_true", help="Silence epoch logs.")
    return p


def main():
    """
    This CLI expects you to provide the model and dataloaders in your own script.
    Typical usage: import this module and call train_rmu(model, forget_loader, retain_loader, **args).
    Keeping main() minimal to avoid imposing dataset/model choices.
    """
    p = build_argparser()
    args = p.parse_args()

    print(
        "This script defines train_rmu(). "
        "Import it and call train_rmu(model, forget_loader, retain_loader, **vars(args))."
    )
    print("Example:")
    print("  from rmu_train import train_rmu")
    print("  model = ...  # your trained CIFAR-10 model")
    print("  forget_loader, retain_loader = ...")
    print("  model = train_rmu(model, forget_loader, retain_loader, **vars(args))")


if __name__ == "__main__":
    main()



# import torch.nn as nn
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import copy
# import numpy as np
# from tqdm import tqdm

# def get_model_layers(model):
#     """
#     Recursively flattens a model into a list of (name, layer) tuples,
#     filtering for specific types of layers that are suitable for
#     sampling activations.
    
#     This works for any model architecture, not just ResNet.
#     """
    
#     # --- Customize this list ---
#     # Define the types of layers you want to be able to sample from.
#     # These are typically the layers that produce the main representations.
#     SAMPLABLE_LAYER_TYPES = (
#         nn.Conv2d,
#         nn.Linear
#     )
    
#     samplable_layers = []
    
#     # model.named_modules() recursively finds EVERY module in the network
#     # (e.g., 'layer1.0.conv1', 'layer1.0.bn1', etc.)
#     for name, module in model.named_modules():
#         # We check if the module is an instance of any of the types
#         # we defined in our tuple above.
#         if isinstance(module, SAMPLABLE_LAYER_TYPES):
#             samplable_layers.append((name, module))
            
#     return samplable_layers

# # def get_model_layers(model):
# #     """
# #     Flattens a sequential or ResNet-like model into a
# #     list of (name, layer) tuples.
# #     """
# #     layers = []
# #     # Example for a simple Sequential model
# #     if isinstance(model, nn.Sequential):
# #         for i, layer in enumerate(model.children()):
# #             layers.append((f"layer_{i}", layer))
# #     # Example for ResNet
# #     elif hasattr(model, 'conv1'):
# #         layers.append(("conv1", model.conv1))
# #         layers.append(("bn1", model.bn1))
# #         layers.append(("relu", model.relu))
# #         layers.append(("maxpool", model.maxpool))
        
# #         if hasattr(model, 'layer1'):
# #             layers.append(("layer1", model.layer1))
# #         if hasattr(model, 'layer2'):
# #             layers.append(("layer2", model.layer2))
# #         if hasattr(model, 'layer3'):
# #             layers.append(("layer3", model.layer3))
# #         if hasattr(model, 'layer4'):
# #             layers.append(("layer4", model.layer4))
            
# #         layers.append(("avgpool", model.avgpool))
# #         layers.append(("fc", model.fc))
# #     else:
# #         # Fallback for other models
# #         for name, layer in model.named_children():
# #             layers.append((name, layer))
            
# #     # We only want to sample from layers that produce activations
# #     # (e.g., Conv, Linear, or blocks) not non-linearities or pooling.
# #     # For this example, let's keep it simple and use all named children.
# #     # In a real case, you would filter this list.
    
# #     return [(name, layer) for name, layer in model.named_children() if 'conv' in name or 'layer' in name or 'fc' in name]


# def get_layer_range_indices(k, num_layers):
#     """
#     Returns the (start, end) indices for layer sampling based on k.
#     """
#     if k == 0.25:
#         # First 25%
#         start_idx = 0
#         end_idx = int(num_layers * 0.25)-1
#     elif k == 0.50:
#         # 20% to 50% (as you specified)
#         start_idx = int(num_layers * 0.25)
#         end_idx = int(num_layers * 0.50)-1
#     elif k == 0.75:
#         # ASSUMPTION: 50% to 75%
#         # print("Assuming k=0.75 samples from 50%-75% of layers.")
#         start_idx = int(num_layers * 0.50)
#         end_idx = int(num_layers * 0.75)-1
#     elif k == 1.0:
#         # ASSUMPTION: 75% to 100%
#         # print("Assuming k=1.0 samples from 75%-100% of layers.")
#         start_idx = int(num_layers * 0.75)
#         end_idx = num_layers - 1 # Use -1 for zero-based index
#     else:
#         raise ValueError(f"Unknown k value: {k}")

#     # Ensure end_idx is at least start_idx and within bounds
#     end_idx = max(start_idx + 1, end_idx) # Ensure at least one layer
#     end_idx = min(end_idx, num_layers - 1)
    
#     return start_idx, end_idx

# def create_random_vectors(model_layers, device='cpu'):
#     """
#     Creates a dictionary of random unit vectors, one for each
#     layer's activation. This is needed because each layer may
#     have a different number of output channels.
#     """
#     u_vectors = {}
#     for name, layer in model_layers:
#         try:
#             # Get output channels for Conv/Linear layers
#             if hasattr(layer, 'out_channels'):
#                 channels = layer.out_channels
#             elif hasattr(layer, 'out_features'):
#                 channels = layer.out_features
#             # Handle ResNet blocks (use the output channels of the last conv)
#             elif isinstance(layer, nn.Sequential):
#                 last_conv = [m for m in layer.modules() if isinstance(m, (nn.Conv2d, nn.Linear))][-1]
#                 channels = last_conv.out_channels if hasattr(last_conv, 'out_channels') else last_conv.out_features
#             else:
#                 # Skip layers we can't get channels for
#                 continue

#             # Create a random vector u
#             u = torch.rand(channels, device=device)
#             # Normalize to make it a unit vector
#             u = u / torch.norm(u)
            
#             # Reshape for broadcasting with (B, C, H, W) activations
#             # New shape: (1, C, 1, 1)
#             u_vectors[name] = u.view(1, -1, 1, 1)
            
#         except Exception as e:
#             print(f"Skipping layer {name}: {e}")
            
#     return u_vectors

# def train_rmu(model, forget_loader, retain_loader, k, epochs=1, lr=1e-3, c=4.0, alpha=1000, device='cpu', seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
    
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     model.to(device)
#     model.train()
    
#     # 1. Create the frozen model
#     model_frozen = copy.deepcopy(model)
#     model_frozen.eval()
#     for param in model_frozen.parameters():
#         param.requires_grad = False
        
#     # 2. Setup optimizer (only for the updated model)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     # 3. Get the list of layers to sample from
#     model_layers = get_model_layers(model) 
#     layer_names = [name for name, _ in model_layers]
#     num_layers = len(layer_names)
    
#     if num_layers == 0:
#         print("Error: get_model_layers() returned 0 samplable layers. Check your SAMPLABLE_LAYER_TYPES.")
#         return model # Return the original model

#     # 4. Create the dictionary of random 'u' vectors
#     u_vectors = create_random_vectors(model_layers, device=device)
    
#     # Store activations from the hook
#     activations = {}
#     def get_activation(name):
#         def hook(model, input, output):
#             activations[name] = output
#         return hook

#     # --- Training Loop ---
#     for epoch in tqdm(range(epochs)):
        
#         # --- 5. Sample ONE layer for the ENTIRE epoch ---
#         start_idx, end_idx = get_layer_range_indices(k, num_layers)
        
#         if start_idx >= end_idx or start_idx >= num_layers or end_idx > num_layers:
#             print(f"Warning: Invalid layer range ({start_idx}, {end_idx}) for k={k} with {num_layers} layers. Skipping epoch {epoch+1}.")
#             continue
            
#         sampled_idx = random.randint(start_idx, end_idx)
#         sampled_layer_name = layer_names[sampled_idx]
        
#         if sampled_layer_name not in u_vectors:
#             print(f"Warning: Layer {sampled_layer_name} has no 'u' vector. Skipping epoch {epoch+1}.")
#             continue

#         # print(f"--- Starting Epoch {epoch+1}/{epochs}, Training on Layer: {sampled_layer_name} ---")

#         # 6. Get the 'u' vector for the sampled layer
#         u = u_vectors[sampled_layer_name].to(device)
        
#         # 7. Register hooks ONCE for the epoch
#         hook_handle_updated = None
#         hook_handle_frozen = None
#         try:
#             hook_handle_updated = model.get_submodule(sampled_layer_name).register_forward_hook(get_activation("updated"))
#             hook_handle_frozen = model_frozen.get_submodule(sampled_layer_name).register_forward_hook(get_activation("frozen"))
#         except AttributeError:
#             print(f"Error: Could not find submodule '{sampled_layer_name}'. Skipping epoch {epoch+1}.")
#             if hook_handle_updated: hook_handle_updated.remove() # Clean up just in case
#             continue
            
#         # --- Batch Loop ---
#         # Iterate over all batches using the *same* layer hook
#         for (forget_data, _) , (retain_data, _) in zip(forget_loader, retain_loader):
#             forget_data = forget_data.to(device)
#             retain_data = retain_data.to(device)
            
#             optimizer.zero_grad()
            
#             # We MUST clear activations dict here so old batch values are gone
#             activations = {} 

#             # --- 8. Calculate Forget Loss ---
#             _ = model(forget_data)
#             act_forget = activations["updated"] # Get activation from hook
            
#             # Handle FC layer (B, C) vs Conv layer (B, C, H, W)
#             u_reshaped = u # Reset u_reshaped for each batch (in case of dim mismatch)
#             if act_forget.dim() == 2: # (Batch, Features)
#                 u_reshaped = u.view(1, -1) # Reshape u to (1, Features)

#             loss_forget = torch.mean((act_forget - c * u_reshaped)**2)

#             # --- 9. Calculate Retain Loss ---
#             # Run frozen model first to populate 'frozen' activation
#             _ = model_frozen(retain_data)
#             act_retain_frozen = activations["frozen"]
            
#             # Run updated model to populate 'updated' activation
#             _ = model(retain_data)
#             act_retain_updated = activations["updated"] # Overwrites forget activation
            
#             loss_retain = torch.mean((act_retain_updated - act_retain_frozen)**2)
            
#             # --- 10. Total Loss and Backpropagation ---
#             loss = loss_forget + alpha * loss_retain
            
#             loss.backward()
#             optimizer.step()
            
#             # (No hook removal here)
            
#         # --- 11. Remove hooks AFTER all batches for the epoch are done ---
#         if hook_handle_updated: hook_handle_updated.remove()
#         if hook_handle_frozen: hook_handle_frozen.remove()
        
#         # print(f"--- Finished Epoch {epoch+1}/{epochs} ---")


#     print("--- RMU Training Complete ---")
    
#     # Return the model that has been unlearned
#     return model
