import torch
import bisect
import matplotlib.pyplot as plt
import numpy as np
import os
import math
if __name__ == '__main__':
    path = '/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/ee_raw.pkl'
    data = torch.load(path)
    min_values = []
    for (key, value) in data['ee_distance_data'].items():
        min_values.append(min(value))
    
    min_values.sort(reverse=False)
    plt.figure(figsize=(4, 6))  # Increase figure size for better readability

    # Create the boxplot
    plt.boxplot(min_values, patch_artist=True, 
                boxprops=dict(facecolor="white", color="black"), 
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5))

    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Label the y-axis
    plt.ylabel('Minimum Distance (cm)', fontsize=12)
    plt.xlabel('')
    # Add a title (optional)
    plt.title("Minimum Distances", fontsize=14)
    plt.savefig(os.path.dirname(path),bbox_inches='tight', dpi=300)
    
    count = 0
    maximum = math.ceil(min_values[-1])
  
    for i in range (0, maximum+1):
        print(f'Less than or equal {i} cm: {100*(bisect.bisect_right(min_values,i)) / len(min_values)}%')