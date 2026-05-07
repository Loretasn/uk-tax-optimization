import pandas as pd
import numpy as np

# Load all ε=1.0 results (10 seeds)
seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
files = [f'/Users/loreta/Desktop/Code_disso/results_eps1.0_seed{s}.csv'  for s in seeds]

baselines = {
    'gross_income': [],
    'gini': [],
    'revenue': [],
    'welfare': [],
    'labour_supply': []
}

for file in files:
    df = pd.read_csv(file)
    
    # Extract UK baseline values
    baselines['gini'].append(df[df['Metric']=='gini']['UK'].values[0])
    baselines['revenue'].append(df[df['Metric']=='revenue']['UK'].values[0])
    baselines['welfare'].append(df[df['Metric']=='welfare']['UK'].values[0])

# Calculate means
print("=" * 60)
print("NEW BASELINE (η = 0.27, 10 runs)")
print("=" * 60)
print(f"Post-tax Gini:        {np.mean(baselines['gini']):.4f}")
print(f"Total Tax Revenue:    £{np.mean(baselines['revenue']):,.0f}")
print(f"Revenue per capita:   £{np.mean(baselines['revenue'])/2000:,.0f}")
print(f"Social Welfare:       {np.mean(baselines['welfare']):,.0f}")
print()

# Standard deviations (should be small if population generation is consistent)
print("Standard deviations across runs:")
print(f"Gini SD:     {np.std(baselines['gini']):.6f}")
print(f"Revenue SD:  £{np.std(baselines['revenue']):,.0f}")
print(f"Welfare SD:  {np.std(baselines['welfare']):.0f}")