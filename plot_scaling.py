import json
import numpy as np
import matplotlib.pyplot as plt

def generate_scaling_plots():
    # 1. Load results written by the training sweep
    if not open("sweep_results.json"):
        print("Error: sweep_results.json not found.")
        return
        
    with open("sweep_results.json", "r") as f:
        data = json.load(f)
        
    param_sweep = data["parameter_sweep"]
    data_sweep = data["data_sweep"]
    
    # 2. Extract values for Parameter Scaling
    N_vals = []
    L_N_vals = []
    for name, res in param_sweep.items():
        N_vals.append(res["N"])
        L_N_vals.append(res["loss"])
        
    log_N = np.log(N_vals)
    log_L_N = np.log(L_N_vals)
    
    # Find slope alpha_N using linear polyfit regression
    slope_N, intercept_N = np.polyfit(log_N, log_L_N, 1)
    alpha_N = -slope_N  # The formula has a minus sign, so we invert it
    
    # 3. Extract values for Data Scaling
    D_vals = []
    L_D_vals = []
    for pct, res in data_sweep.items():
        D_vals.append(res["D"])
        L_D_vals.append(res["loss"])
        
    log_D = np.log(D_vals)
    log_L_D = np.log(L_D_vals)
    
    # Find slope alpha_D using linear polyfit regression
    slope_D, intercept_D = np.polyfit(log_D, log_L_D, 1)
    alpha_D = -slope_D
    
    # Calculate the ultimate scaling ratio
    gamma = alpha_N / alpha_D if alpha_D != 0 else 0
    
    print("--- Empirical Scaling Law Metrics Fit ---")
    print(f"Calculated Alpha_N (Parameter Exponent): {alpha_N:.4f}")
    print(f"Calculated Alpha_D (Data Exponent):      {alpha_D:.4f}")
    print(f"Calculated Scaling Ratio (Gamma):        {gamma:.4f}")
    
    # 4. Generate the required log-log validation scaling plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Parameter scaling line fit
    ax1.scatter(N_vals, L_N_vals, color='red', label='Experimental Data', zorder=5)
    fit_N_loss = np.exp(intercept_N) * (np.array(N_vals) ** slope_N)
    ax1.plot(N_vals, fit_N_loss, color='blue', linestyle='--', label=f'Fit Line (α_N={alpha_N:.3f})')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Non-Embedding Parameters (N)')
    ax1.set_ylabel('Validation Loss (L)')
    ax1.set_title('Loss vs Model Parameters (N)')
    ax1.grid(True, which="both", ls="-")
    ax1.legend()
    
    # Right subplot: Data scaling line fit
    ax2.scatter(D_vals, L_D_vals, color='orange', label='Experimental Data', zorder=5)
    fit_D_loss = np.exp(intercept_D) * (np.array(D_vals) ** slope_D)
    ax2.plot(D_vals, fit_D_loss, color='green', linestyle='--', label=f'Fit Line (α_D={alpha_D:.3f})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Dataset Size in Tokens (D)')
    ax2.set_ylabel('Validation Loss (L)')
    ax2.set_title('Loss vs Training Dataset Size (D)')
    ax2.grid(True, which="both", ls="-")
    ax2.legend()
    
    plt.suptitle(f"Empirical Scaling Laws (Ratio Gamma = {gamma:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig("scaling_laws.png")
    plt.show()

if __name__ == "__main__":
    generate_scaling_plots()
