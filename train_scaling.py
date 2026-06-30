import torch
import numpy as np
from models import DecoderTransformer

def get_batch(data, batch_size, block_size, device):
    """
    Custom data pipeline to pack raw tokens into batches of X, Y.
    """
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def count_non_embedding_params(model):
    """
    Programmatic validation excluding token embeddings and language model heads.
    """
    return sum(p.numel() for name, p in model.named_parameters() 
               if 'token_embedding' not in name and 'position_embedding' not in name and 'lm_head' not in name)

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters, device):
    """
    Tracks validation loss limits over controlled evaluations[cite: 49, 54].
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

import os
import json

def run_sweeps(train_data, val_data, vocab_size, device, is_test_run=True):
    # If is_test_run is True, we do a quick check of 100 steps. 
    # Change to False for the final 3000 step run required by your assignment!
    total_steps = 100 if is_test_run else 3000
    eval_interval = 20 if is_test_run else 200
    eval_iters = 5 if is_test_run else 50
    
    # Static global configurations
    block_size = 128
    batch_size = 32
    learning_rate = 1e-3
    
    print(f"Launching Sweeps (Mode: {'TEST RUN' if is_test_run else 'OFFICIAL SWEEP'}) ---")
    
    
    # SWEEP 1: Parameter Scaling Sweep
    
    print("\n Starting Sweep 1: Parameter Scaling ")
    param_configs = {
        "Tiny":   {"layers": 2, "d_model": 64,  "n_heads": 2, "d_ff": 256},
        "Small":  {"layers": 4, "d_model": 128, "n_heads": 4, "d_ff": 512},
        "Medium": {"layers": 6, "d_model": 256, "n_heads": 8, "d_ff": 1024}
    }
    
    parameter_sweep_results = {}
    
    for name, config in param_configs.items():
        print(f"Training {name} model scale...")
        model = DecoderTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            n_layers=config["layers"],
            max_seq_len=block_size
        ).to(device)
        
        N = count_non_embedding_params(model)
        print(f"Model {name} counted non-embedding parameters N = {N}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        
        for step in range(total_steps):
            x, y = get_batch(train_data, batch_size, block_size, device)
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % eval_interval == 0 or step == total_steps - 1:
                val_loss = estimate_loss(model, val_data, batch_size, block_size, eval_iters, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"  Step {step}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                
        parameter_sweep_results[name] = {"N": N, "loss": best_val_loss}
        
    
    # SWEEP 2: Data Scaling Sweep
    
    print("\n Starting Sweep 2: Data Scaling ")
    data_percentages = [0.10, 0.25, 0.50, 1.00]
    data_sweep_results = {}
    
    small_cfg = param_configs["Small"]
    
    for p in data_percentages:
        print(f"Training Small model on {int(p*100)}% of token dataset...")
        
        # Take a clean slice of the training tokens
        tokens_cutoff = int(p * len(train_data))
        subset_train_data = train_data[:tokens_cutoff]
        
        model = DecoderTransformer(
            vocab_size=vocab_size,
            d_model=small_cfg["d_model"],
            n_heads=small_cfg["n_heads"],
            d_ff=small_cfg["d_ff"],
            n_layers=small_cfg["layers"],
            max_seq_len=block_size
        ).to(device)
        
        D = tokens_cutoff
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        
        for step in range(total_steps):
            x, y = get_batch(subset_train_data, batch_size, block_size, device)
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % eval_interval == 0 or step == total_steps - 1:
                val_loss = estimate_loss(model, val_data, batch_size, block_size, eval_iters, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"  Step {step}/{total_steps} | Val Loss: {val_loss:.4f}")
                
        data_sweep_results[int(p*100)] = {"D": D, "loss": best_val_loss}
        
    # Save results to a json file to be read by the plotting script
    sweep_output = {
        "parameter_sweep": parameter_sweep_results,
        "data_sweep": data_sweep_results
    }
    
    with open("sweep_results.json", "w") as f:
        json.dump(sweep_output, f, indent=4)
    print("\n[SUCCESS] Sweeps complete. Data written to sweep_results.json")
