Optimal Income Tax Design for the UK

Replication code for BSc dissertation (Queen Mary University of London, 2026).

Author: Loreta Sneidere  
Supervisor: Dr Hao Ma

---
Summary

Agent-based genetic algorithm optimization of UK income tax structure under revenue neutrality. 

Finding: Current UK system inefficient - welfare gains of 1.7-2.0% achievable through threshold compression and allowance elevation.

---
Files

- src/tax_env.py - Agent-based tax environment (2000 agents, ONS calibrated)
- src/sensitivity_analysis.py - Genetic algorithm optimization
- results/all_results_combined.csv - 40 optimization runs (10 seeds × 4 ε values)
- results/epsilon*.csv - UK baseline for each inequality aversion level

---
Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Test environment
python src/tax_env.py

# Run optimization (single ε)
python src/sensitivity_analysis.py --epsilon 1.2 --seeds 1

# Full sensitivity analysis (reproduces dissertation results, ~80 hours)
python src/sensitivity_analysis.py --full-sensitivity --seeds 10
```

---
Requirements

Python 3.9+, NumPy, SciPy, Pandas, Matplotlib

---
Data Sources

- Income distribution: ONS "Effects of taxes and benefits on household income" (2024), Table 13
- Tax parameters: HMRC Income Tax rates 2024/25

---

Citation

Sneidere, L. (2026). Optimal Income Tax Design Under Revenue Constraints:
An Agent-Based Genetic Algorithm Analysis of the UK Tax System.
BSc Dissertation, Queen Mary University of London.


MIT license
