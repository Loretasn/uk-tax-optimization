"""
UK TAX OPTIMIZATION - COMPLETE SENSITIVITY & ROBUSTNESS ANALYSIS
PRODUCTION-READY VERSION
"""

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

import sys
import os

print("=" * 80)
print("CHECKING DEPENDENCIES...")
print("=" * 80)

if sys.version_info < (3, 7):
    print("ERROR: Python 3.7+ required")
    print(f"Your version: {sys.version}")
    sys.exit(1)
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")

required_packages = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"{name} found")
    except ImportError:
        missing_packages.append(name)
        print(f"{name} NOT found")

if missing_packages:
    print("\nERROR: Missing required packages:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nInstall with: pip install numpy pandas matplotlib")
    sys.exit(1)

if not os.path.exists("tax_env.py"):
    print("\nERROR: tax_env.py not found in current directory")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)
print("tax_env.py found")

print("\nAll dependencies satisfied!")
print("=" * 80)
print()

# ============================================================================
# IMPORTS
# ============================================================================

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import time
import traceback
import json

try:
    from tax_env import TaxEnvironment
    print("TaxEnvironment imported successfully")
except Exception as e:
    print(f"\nERROR importing TaxEnvironment: {str(e)}")
    sys.exit(1)

print()

# ============================================================================
# CONFIGURATION
# ============================================================================

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print()

# Options:
# "MINIMAL"   -> 1 seed per epsilon
# "MODERATE"  -> 5 seeds per epsilon
# "RIGOROUS"  -> 10 seeds per epsilon

TIME_BUDGET = "RIGOROUS"   # CHANGE IF NEEDED

EPSILONS = [1.0, 1.2, 1.5, 2.0]

if TIME_BUDGET == "MINIMAL":
    SEEDS = [42]
    print("MINIMAL mode: 1 seed per epsilon")
elif TIME_BUDGET == "MODERATE":
    SEEDS = [42, 101, 202, 303, 404]
    print("MODERATE mode: 5 seeds per epsilon")
elif TIME_BUDGET == "RIGOROUS":
    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    print("RIGOROUS mode: 10 seeds per epsilon")
else:
    print(f"Unknown TIME_BUDGET '{TIME_BUDGET}', defaulting to MINIMAL")
    SEEDS = [42]
    TIME_BUDGET = "MINIMAL"

N_AGENTS = 2000
POPULATION_SIZE = 40
N_GENERATIONS = 60

total_runs = len(EPSILONS) * len(SEEDS)
estimated_minutes_per_run = 30
estimated_hours = total_runs * estimated_minutes_per_run / 60

print("\nSettings:")
print(f"  Epsilon values: {EPSILONS}")
print(f"  Random seeds: {SEEDS}")
print(f"  Agents per run: {N_AGENTS:,}")
print(f"  GA population: {POPULATION_SIZE}")
print(f"  GA generations: {N_GENERATIONS}")
print(f"\nTotal runs: {len(EPSILONS)} x {len(SEEDS)} = {total_runs}")
print(f"Estimated time: ~{estimated_hours:.1f} hours")
print()

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

CHECKPOINT_FILE = "optimization_checkpoint.json"

def save_checkpoint(completed_runs):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed_runs": completed_runs,
            "time_budget": TIME_BUDGET,
            "epsilons": EPSILONS,
            "seeds": SEEDS,
        }, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            if (
                data["time_budget"] == TIME_BUDGET
                and data["epsilons"] == EPSILONS
                and data["seeds"] == SEEDS
            ):
                return data["completed_runs"]
            else:
                print("Found checkpoint but configuration changed - starting fresh")
                return []
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            return []
    return []

# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class SimpleGA:
    def __init__(self, population_size=40):
        self.population_size = population_size
        self.mutation_rate = 0.15
        self.bounds_low = np.array([12000, 30000, 60000, 0.10, 0.25, 0.40])
        self.bounds_high = np.array([20000, 60000, 130000, 0.24, 0.44, 0.58])
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            policy = np.random.uniform(self.bounds_low, self.bounds_high)
            policy = self._enforce_constraints(policy)
            population.append(policy)
        return np.array(population)

    def _enforce_constraints(self, policy):
        try:
            policy = np.clip(policy, self.bounds_low, self.bounds_high)
            policy[1] = max(policy[1], policy[0] + 5000)
            policy[2] = max(policy[2], policy[1] + 10000)
            policy[4] = max(policy[4], policy[3] + 0.05)
            policy[5] = max(policy[5], policy[4] + 0.05)
            return policy
        except Exception:
            return np.array([12570, 35000, 80000, 0.15, 0.35, 0.45])

    def evaluate_fitness(self, env, revenue_target_abs, revenue_min, revenue_max, uk_gini):
        fitness = np.zeros(self.population_size)

        for i, policy in enumerate(self.population):
            try:
                results = env.simulate(policy)

                fitness[i] = results["social_welfare"]

                revenue = results["revenue"]
                if revenue < revenue_min:
                    shortfall_share = (revenue_min - revenue) / revenue_target_abs
                    fitness[i] -= 20000 * shortfall_share
                elif revenue > revenue_max:
                    excess_share = (revenue - revenue_max) / revenue_target_abs
                    fitness[i] -= 5000 * excess_share
                else:
                    proximity = 1 - abs(revenue - revenue_target_abs) / revenue_target_abs
                    fitness[i] += 100 * proximity

                gini = results["gini"]
                if gini < uk_gini:
                    improvement_share = (uk_gini - gini) / uk_gini
                    fitness[i] += 200 * improvement_share
                else:
                    worsening_share = (gini - uk_gini) / uk_gini
                    fitness[i] -= 300 * worsening_share

            except Exception as e:
                fitness[i] = -1e10
                if i == 0:
                    print(f"    Simulation error (policy {i}): {str(e)[:80]}")

        return fitness

    def evolve(self, fitness):
        try:
            best_idx = np.argmax(fitness)
            best_policy = self.population[best_idx].copy()
            best_fitness = fitness[best_idx]

            n_parents = max(12, self.population_size // 3)
            sorted_indices = np.argsort(fitness)[::-1]
            parents = self.population[sorted_indices[:n_parents]]

            new_population = [best_policy.copy()]

            while len(new_population) < self.population_size:
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                mask = np.random.rand(len(p1)) > 0.5
                child = np.where(mask, p1, p2)

                for i in range(len(child)):
                    if np.random.rand() < self.mutation_rate:
                        noise = np.random.normal(
                            0, 0.05 * (self.bounds_high[i] - self.bounds_low[i])
                        )
                        child[i] += noise

                child = self._enforce_constraints(child)
                new_population.append(child)

            self.population = np.array(new_population)
            return best_policy, best_fitness

        except Exception as e:
            print(f"    Evolution error: {e}")
            best_idx = np.argmax(fitness)
            return self.population[best_idx].copy(), fitness[best_idx]

# ============================================================================
# MAIN LOOP
# ============================================================================

print("=" * 80)
print("STARTING OPTIMIZATION RUNS")
print("=" * 80)
print()

completed_runs = load_checkpoint()
if completed_runs:
    print(f"Found {len(completed_runs)} completed runs - will skip those")
    print()

all_results = []
overall_start = time.time()
current_run = 0
successful_runs = 0
failed_runs = []

for epsilon in EPSILONS:
    for seed in SEEDS:
        current_run += 1
        run_id = f"eps{epsilon}_seed{seed}"

        if run_id in completed_runs:
            print(f"Skipping {current_run}/{total_runs}: ε={epsilon}, seed={seed} (already completed)")
            continue

        run_start = time.time()

        print("\n" + "=" * 80)
        print(f"RUN {current_run}/{total_runs}: ε={epsilon}, seed={seed}")
        print("=" * 80)

        try:
            np.random.seed(seed)
            print(f"Random seed set to {seed}")

            env = TaxEnvironment(n_agents=N_AGENTS, equity_weight=0.7)
            env.epsilon_swf = epsilon
            print(f"Environment created ({N_AGENTS:,} agents, ε={epsilon})")

            print("Calculating UK baseline...")
            uk_results = env.benchmark_uk_system()
            revenue_target_absolute = uk_results["revenue"]
            revenue_target_min = revenue_target_absolute * 0.95
            revenue_target_max = revenue_target_absolute * 1.05

            print(f"  Gini: {uk_results['gini']:.4f}")
            print(f"  Revenue: £{uk_results['revenue']:,.0f}")
            print(f"  Social Welfare: {uk_results['social_welfare']:.2f}")
            print(
                f"  Revenue Target Range: £{revenue_target_min:,.0f} to £{revenue_target_max:,.0f} (-5% / +10%)"
            )

            print(f"\nInitializing GA (pop={POPULATION_SIZE}, gen={N_GENERATIONS})...")
            ga = SimpleGA(population_size=POPULATION_SIZE)
            best_policy = None
            best_fitness = -np.inf

            print("Training:")
            for gen in range(N_GENERATIONS):
                fitness = ga.evaluate_fitness(
                    env,
                    revenue_target_absolute,
                    revenue_target_min,
                    revenue_target_max,
                    uk_results["gini"],
                )

                best_policy, best_fitness = ga.evolve(fitness)
                ga.fitness_history.append(best_fitness)

                if (gen + 1) % 10 == 0 or gen == 0:
                    print(f"  Gen {gen+1:2d}/{N_GENERATIONS}: Best fitness = {best_fitness:,.2f}")

            print(f"Training complete! Best fitness: {best_fitness:,.2f}")

            print("\nFinal evaluation...")
            optimal_results = env.simulate(best_policy)

            gini_change = ((optimal_results["gini"] - uk_results["gini"]) / uk_results["gini"]) * 100
            revenue_change = ((optimal_results["revenue"] - uk_results["revenue"]) / uk_results["revenue"]) * 100
            welfare_change = ((optimal_results["social_welfare"] - uk_results["social_welfare"]) / uk_results["social_welfare"]) * 100

            print("\n" + "=" * 80)
            print(f"RESULTS: ε={epsilon}, seed={seed}")
            print("=" * 80)
            print("Optimal Policy:")
            print(f"  Allowance:        £{best_policy[0]:>9,.0f}")
            print(f"  Basic Threshold:  £{best_policy[1]:>9,.0f}")
            print(f"  Higher Threshold: £{best_policy[2]:>9,.0f}")
            print(f"  Basic Rate:       {best_policy[3]*100:>9.2f}%")
            print(f"  Higher Rate:      {best_policy[4]*100:>9.2f}%")
            print(f"  Additional Rate:  {best_policy[5]*100:>9.2f}%")
            print("\nOutcomes:")
            print(f"  Gini:    {uk_results['gini']:.4f} → {optimal_results['gini']:.4f} ({gini_change:+.2f}%)")
            print(f"  Revenue: £{uk_results['revenue']/1e6:.2f}M → £{optimal_results['revenue']/1e6:.2f}M ({revenue_change:+.2f}%)")
            print(f"  Welfare: {uk_results['social_welfare']:.2f} → {optimal_results['social_welfare']:.2f} ({welfare_change:+.2f}%)")

            result = {
                "epsilon": epsilon,
                "seed": seed,
                "uk_gini": uk_results["gini"],
                "uk_revenue": uk_results["revenue"],
                "uk_welfare": uk_results["social_welfare"],
                "opt_gini": optimal_results["gini"],
                "opt_revenue": optimal_results["revenue"],
                "opt_welfare": optimal_results["social_welfare"],
                "gini_change_pct": gini_change,
                "revenue_change_pct": revenue_change,
                "welfare_change_pct": welfare_change,
                "allowance": best_policy[0],
                "basic_threshold": best_policy[1],
                "higher_threshold": best_policy[2],
                "basic_rate": best_policy[3],
                "higher_rate": best_policy[4],
                "additional_rate": best_policy[5],
            }
            all_results.append(result)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epsilon_str = str(epsilon).replace(".", "_")
            csv_file = f"results_eps{epsilon_str}_seed{seed}_{timestamp}.csv"

            result_df = pd.DataFrame({
                "Metric": [
                    "Epsilon",
                    "Seed",
                    "Gini",
                    "Revenue",
                    "Atkinson_Social_Welfare",
                    "Allowance",
                    "Basic_Threshold",
                    "Higher_Threshold",
                    "Basic_Rate",
                    "Higher_Rate",
                    "Top_Rate",
                ],
                "UK": [
                    epsilon,
                    seed,
                    uk_results["gini"],
                    uk_results["revenue"],
                    uk_results["social_welfare"],
                    12570,
                    50270,
                    125140,
                    0.20,
                    0.40,
                    0.45,
                ],
                "Optimal": [
                    epsilon,
                    seed,
                    optimal_results["gini"],
                    optimal_results["revenue"],
                    optimal_results["social_welfare"],
                    best_policy[0],
                    best_policy[1],
                    best_policy[2],
                    best_policy[3],
                    best_policy[4],
                    best_policy[5],
                ],
            })
            result_df.to_csv(csv_file, index=False)
            print(f"\nSaved: {csv_file}")

            completed_runs.append(run_id)
            save_checkpoint(completed_runs)

            successful_runs += 1

            run_time = (time.time() - run_start) / 60
            elapsed_total = (time.time() - overall_start) / 60
            completed_count = max(successful_runs + len(failed_runs), 1)
            avg_time_per_run = elapsed_total / completed_count
            remaining_runs = total_runs - (successful_runs + len(failed_runs))
            estimated_remaining = remaining_runs * avg_time_per_run

            print(f"\nRun completed in {run_time:.1f} minutes")
            print(f"Progress: {successful_runs + len(failed_runs)}/{total_runs} ({(successful_runs + len(failed_runs))/total_runs*100:.1f}%)")
            print(f"Elapsed: {elapsed_total:.1f} min ({elapsed_total/60:.1f} hours)")
            print(f"Estimated remaining: {estimated_remaining:.1f} min ({estimated_remaining/60:.1f} hours)")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            print(f"Progress saved. {successful_runs} runs completed.")
            sys.exit(0)

        except Exception as e:
            print(f"\nERROR in run {current_run}/{total_runs} (ε={epsilon}, seed={seed}):")
            print(str(e))
            print("\nFull traceback:")
            traceback.print_exc()

            failed_runs.append({
                "run": current_run,
                "epsilon": epsilon,
                "seed": seed,
                "error": str(e),
            })

            print("\nContinuing with next run...")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("GENERATING SUMMARY")
print("=" * 80)
print()

if not all_results:
    print("No successful runs to summarize.")
    if failed_runs:
        print(f"\n{len(failed_runs)} runs failed:")
        for fail in failed_runs:
            print(f"  Run {fail['run']}: ε={fail['epsilon']}, seed={fail['seed']}")
            print(f"  Error: {fail['error']}")
    sys.exit(1)

df_all = pd.DataFrame(all_results)
df_all.to_csv("all_results_combined.csv", index=False)
print("Saved: all_results_combined.csv")

if len(SEEDS) > 1:
    summary = df_all.groupby("epsilon").agg({
        "allowance": ["mean", "std"],
        "basic_threshold": ["mean", "std", "min", "max", "count"],
        "higher_threshold": ["mean", "std"],
        "basic_rate": ["mean", "std"],
        "higher_rate": ["mean", "std"],
        "additional_rate": ["mean", "std"],
        "welfare_change_pct": ["mean", "std"],
        "revenue_change_pct": ["mean", "std"],
        "gini_change_pct": ["mean", "std"],
    })

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Multiple Seeds)")
    print("=" * 80)
    print(summary)

    cv_data = []
    for eps in EPSILONS:
        eps_data = df_all[df_all["epsilon"] == eps]
        if len(eps_data) == 0:
            continue

        mean_threshold = eps_data["basic_threshold"].mean()
        std_threshold = eps_data["basic_threshold"].std()
        mean_basic_rate = eps_data["basic_rate"].mean()
        std_basic_rate = eps_data["basic_rate"].std()
        mean_welfare = eps_data["welfare_change_pct"].mean()
        std_welfare = eps_data["welfare_change_pct"].std()

        cv_threshold = (std_threshold / mean_threshold * 100) if mean_threshold != 0 else 0
        cv_basic_rate = (std_basic_rate / mean_basic_rate * 100) if mean_basic_rate != 0 else 0
        cv_welfare = (std_welfare / abs(mean_welfare) * 100) if mean_welfare != 0 else 0

        cv_data.append({
            "epsilon": eps,
            "n_runs": len(eps_data),
            "threshold_cv_pct": cv_threshold,
            "basic_rate_cv_pct": cv_basic_rate,
            "welfare_cv_pct": cv_welfare,
        })

    cv_df = pd.DataFrame(cv_data)
    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK")
    print("=" * 80)
    print(cv_df.to_string(index=False))

else:
    cv_df = pd.DataFrame()

# ============================================================================
# DISSERTATION TABLE
# ============================================================================

dissertation_table = []

for eps in EPSILONS:
    eps_data = df_all[df_all["epsilon"] == eps]
    if len(eps_data) == 0:
        continue

    row = {
        "Epsilon": eps,
        "Allowance": f"£{eps_data['allowance'].mean():,.0f}",
        "Basic_Threshold": f"£{eps_data['basic_threshold'].mean():,.0f}",
        "Higher_Threshold": f"£{eps_data['higher_threshold'].mean():,.0f}",
        "Basic_Rate": f"{eps_data['basic_rate'].mean()*100:.2f}%",
        "Higher_Rate": f"{eps_data['higher_rate'].mean()*100:.2f}%",
        "Additional_Rate": f"{eps_data['additional_rate'].mean()*100:.2f}%",
        "Gini_Change": f"{eps_data['gini_change_pct'].mean():+.2f}%",
        "Revenue_Change": f"{eps_data['revenue_change_pct'].mean():+.2f}%",
        "Welfare_Change": f"{eps_data['welfare_change_pct'].mean():+.2f}%",
        "Runs": len(eps_data),
    }

    if len(eps_data) > 1:
        row["Threshold_SD"] = f"±£{eps_data['basic_threshold'].std():,.0f}"
        row["Welfare_SD"] = f"±{eps_data['welfare_change_pct'].std():.2f}"
        if not cv_df.empty:
            cv = cv_df[cv_df["epsilon"] == eps]["threshold_cv_pct"].iloc[0]
            row["Threshold_CV"] = f"{cv:.2f}%"

    dissertation_table.append(row)

dissertation_df = pd.DataFrame(dissertation_table)

print("\n" + "=" * 80)
print("TABLE FOR DISSERTATION")
print("=" * 80)
print(dissertation_df.to_string(index=False))

dissertation_df.to_csv("summary_for_dissertation.csv", index=False)
print("\nSaved: summary_for_dissertation.csv")

# ============================================================================
# OPTIONAL PLOT
# ============================================================================

try:
    plt.figure(figsize=(10, 6))
    for eps in EPSILONS:
        eps_data = df_all[df_all["epsilon"] == eps]
        if len(eps_data) > 0:
            plt.scatter(
                [eps] * len(eps_data),
                eps_data["basic_threshold"],
                alpha=0.7,
                label=f"ε={eps}"
            )

    plt.xlabel("Epsilon")
    plt.ylabel("Optimal Basic Threshold (£)")
    plt.title("Optimal Basic Threshold by Inequality Aversion")
    plt.grid(True, alpha=0.3)
    plt.savefig("threshold_by_epsilon.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: threshold_by_epsilon.png")
except Exception as e:
    print(f"Could not create plot: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = (time.time() - overall_start) / 3600

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"Successful runs: {successful_runs}/{total_runs}")
if failed_runs:
    print(f"Failed runs: {len(failed_runs)}/{total_runs}")
print(f"Total time: {total_time:.2f} hours")
print(f"Average per successful run: {total_time/max(successful_runs, 1)*60:.1f} minutes")

print("\nFiles created:")
print(f"  - {successful_runs} individual CSV files")
print("  - all_results_combined.csv")
print("  - summary_for_dissertation.csv")
print("  - threshold_by_epsilon.png")

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("  - checkpoint file cleaned up")

print("\nAll optimization complete. Ready for dissertation.")
print("=" * 80)