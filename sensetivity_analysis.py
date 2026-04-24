import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

from tax_env import TaxEnvironment

# config
TIME_BUDGET = "RIGOROUS"
EPSILONS = [1.0, 1.2, 1.5, 2.0]

if TIME_BUDGET == "MINIMAL":
    SEEDS = [42]
elif TIME_BUDGET == "MODERATE":
    SEEDS = [42, 101, 202, 303, 404]
elif TIME_BUDGET == "RIGOROUS":
    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
else:
    SEEDS = [42]

N_AGENTS = 2000
POPULATION_SIZE = 40
N_GENERATIONS = 60

print(f"Running {len(EPSILONS)} x {len(SEEDS)} = {len(EPSILONS)*len(SEEDS)} optimizations")
print(f"Estimated time: ~{len(EPSILONS)*len(SEEDS)*0.5:.1f} hours\n")

CHECKPOINT_FILE = "optimization_checkpoint.json"

def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed": completed, "config": TIME_BUDGET}, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
            if data["config"] == TIME_BUDGET:
                return data["completed"]
        except:
            pass
    return []


class SimpleGA:
    def __init__(self, pop_size=40):
        self.pop_size = pop_size
        self.mutation = 0.15
        self.low = np.array([12000, 30000, 60000, 0.10, 0.25, 0.40])
        self.high = np.array([20000, 60000, 130000, 0.24, 0.44, 0.58])
        self.population = self._init_pop()
        self.history = []

    def _init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            policy = np.random.uniform(self.low, self.high)
            policy = self._fix_constraints(policy)
            pop.append(policy)
        return np.array(pop)

    def _fix_constraints(self, p):
        p = np.clip(p, self.low, self.high)
        p[1] = max(p[1], p[0] + 5000)
        p[2] = max(p[2], p[1] + 10000)
        p[4] = max(p[4], p[3] + 0.05)
        p[5] = max(p[5], p[4] + 0.05)
        return p

    def eval_fitness(self, env, rev_target, rev_min, rev_max, uk_gini):
        fitness = np.zeros(self.pop_size)
        
        for i, policy in enumerate(self.population):
            try:
                res = env.simulate(policy)
                fitness[i] = res["social_welfare"]
                
                # revenue penalty
                rev = res["revenue"]
                if rev < rev_min:
                    fitness[i] -= 20000 * (rev_min - rev) / rev_target
                elif rev > rev_max:
                    fitness[i] -= 5000 * (rev - rev_max) / rev_target
                else:
                    fitness[i] += 100 * (1 - abs(rev - rev_target) / rev_target)
                
                # gini adjustment
                gini = res["gini"]
                if gini < uk_gini:
                    fitness[i] += 200 * (uk_gini - gini) / uk_gini
                else:
                    fitness[i] -= 300 * (gini - uk_gini) / uk_gini
                    
            except:
                fitness[i] = -1e10
                
        return fitness

    def evolve(self, fitness):
        best_idx = np.argmax(fitness)
        best = self.population[best_idx].copy()
        
        # selection
        sorted_idx = np.argsort(fitness)[::-1]
        parents_idx = sorted_idx[:self.pop_size//2]
        parents = self.population[parents_idx]
        
        # crossover
        offspring = []
        for _ in range(self.pop_size//2):
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            child = np.where(np.random.rand(6) < 0.5, p1, p2)
            offspring.append(child)
        offspring = np.array(offspring)
        
        # mutation
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation:
                param = np.random.randint(6)
                if param < 3:
                    offspring[i][param] += np.random.normal(0, 1000)
                else:
                    offspring[i][param] += np.random.normal(0, 0.01)
                offspring[i] = self._fix_constraints(offspring[i])
        
        # combine
        self.population = np.vstack([parents, offspring])
        self.population[0] = best  # elitism
        
        return best, fitness[best_idx]

    def run(self, env, uk_revenue, uk_gini, generations=60):
        rev_target = uk_revenue
        rev_min = uk_revenue * 0.95
        rev_max = uk_revenue * 1.05
        
        best_policy = None
        best_fitness = -np.inf
        
        for gen in range(generations):
            fitness = self.eval_fitness(env, rev_target, rev_min, rev_max, uk_gini)
            policy, fit = self.evolve(fitness)
            
            if fit > best_fitness:
                best_fitness = fit
                best_policy = policy.copy()
            
            self.history.append(fit)
            
            if gen % 10 == 0 or gen == generations - 1:
                print(f"  Gen {gen:2d}: fitness={fit:,.0f}")
        
        return best_policy


# main loop
overall_start = time.time()
all_results = []
failed = []
completed = load_checkpoint()

current = 0
total = len(EPSILONS) * len(SEEDS)

for epsilon in EPSILONS:
    for seed in SEEDS:
        current += 1
        run_id = f"eps{epsilon}_seed{seed}"
        
        if run_id in completed:
            print(f"\n[{current}/{total}] Skipping {run_id} (already done)")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{current}/{total}] Starting: epsilon={epsilon}, seed={seed}")
        print(f"{'='*60}")
        
        run_start = time.time()
        
        try:
            np.random.seed(seed)
            
            # create env
            env = TaxEnvironment(n_agents=N_AGENTS)
            env.epsilon_swf = epsilon
            
            # uk baseline
            print("Running UK baseline...")
            uk = env.benchmark_uk_system()
            print(f"UK: Gini={uk['gini']:.4f}, Revenue=£{uk['revenue']:,.0f}, Welfare={uk['social_welfare']:.2f}")
            
            # optimize
            print(f"\nOptimizing with GA...")
            ga = SimpleGA(pop_size=POPULATION_SIZE)
            best = ga.run(env, uk['revenue'], uk['gini'], generations=N_GENERATIONS)
            
            # eval best
            opt = env.simulate(best)
            print(f"\nOptimal: Gini={opt['gini']:.4f}, Revenue=£{opt['revenue']:,.0f}, Welfare={opt['social_welfare']:.2f}")
            
            welfare_change = ((opt['social_welfare'] - uk['social_welfare']) / uk['social_welfare']) * 100
            revenue_change = ((opt['revenue'] - uk['revenue']) / uk['revenue']) * 100
            gini_change = ((opt['gini'] - uk['gini']) / uk['gini']) * 100
            
            print(f"\nChanges: Welfare={welfare_change:+.2f}%, Revenue={revenue_change:+.2f}%, Gini={gini_change:+.2f}%")
            
            # save result
            result = {
                "epsilon": epsilon,
                "seed": seed,
                "allowance": best[0],
                "basic_threshold": best[1],
                "higher_threshold": best[2],
                "basic_rate": best[3],
                "higher_rate": best[4],
                "additional_rate": best[5],
                "gini": opt['gini'],
                "revenue": opt['revenue'],
                "welfare": opt['social_welfare'],
                "uk_gini": uk['gini'],
                "uk_revenue": uk['revenue'],
                "uk_welfare": uk['social_welfare'],
                "welfare_change_pct": welfare_change,
                "revenue_change_pct": revenue_change,
                "gini_change_pct": gini_change,
            }
            all_results.append(result)
            
            # save individual csv
            csv_name = f"results_eps{epsilon}_seed{seed}.csv"
            df = pd.DataFrame({
                "Metric": ["epsilon", "seed", "gini", "revenue", "welfare", "allowance", "basic_threshold", "higher_threshold", "basic_rate", "higher_rate", "additional_rate"],
                "UK": [epsilon, seed, uk['gini'], uk['revenue'], uk['social_welfare'], 12570, 50270, 125140, 0.20, 0.40, 0.45],
                "Optimal": [epsilon, seed, opt['gini'], opt['revenue'], opt['social_welfare'], best[0], best[1], best[2], best[3], best[4], best[5]],
            })
            df.to_csv(csv_name, index=False)
            
            completed.append(run_id)
            save_checkpoint(completed)
            
            runtime = (time.time() - run_start) / 60
            elapsed = (time.time() - overall_start) / 60
            remaining = (total - current) * (elapsed / current)
            
            print(f"\nCompleted in {runtime:.1f} min")
            print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} hrs, Remaining: ~{remaining/60:.1f} hrs")
            
        except Exception as e:
            print(f"\nError: {e}")
            failed.append({"epsilon": epsilon, "seed": seed, "error": str(e)})


# summary
print(f"\n{'='*60}")
print("Creating summary...")
print(f"{'='*60}\n")

if all_results:
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("all_results_combined.csv", index=False)
    print("Saved: all_results_combined.csv")
    
    # dissertation table
    summary = []
    for eps in EPSILONS:
        data = df_all[df_all["epsilon"] == eps]
        if len(data) > 0:
            summary.append({
                "Epsilon": eps,
                "Allowance": f"£{data['allowance'].mean():,.0f}",
                "Basic_Threshold": f"£{data['basic_threshold'].mean():,.0f}",
                "Higher_Threshold": f"£{data['higher_threshold'].mean():,.0f}",
                "Basic_Rate": f"{data['basic_rate'].mean()*100:.1f}%",
                "Higher_Rate": f"{data['higher_rate'].mean()*100:.1f}%",
                "Additional_Rate": f"{data['additional_rate'].mean()*100:.1f}%",
                "Welfare_Gain": f"{data['welfare_change_pct'].mean():+.2f}%",
                "Revenue_Change": f"{data['revenue_change_pct'].mean():+.2f}%",
                "Runs": len(data),
            })
    
    summary_df = pd.DataFrame(summary)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("summary_for_dissertation.csv", index=False)
    
    # plot
    try:
        plt.figure(figsize=(10, 6))
        for eps in EPSILONS:
            data = df_all[df_all["epsilon"] == eps]
            if len(data) > 0:
                plt.scatter([eps]*len(data), data["basic_threshold"], alpha=0.6, label=f"ε={eps}")
        plt.xlabel("Epsilon")
        plt.ylabel("Basic Threshold (£)")
        plt.title("Optimal Threshold by Inequality Aversion")
        plt.grid(alpha=0.3)
        plt.savefig("threshold_by_epsilon.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: threshold_by_epsilon.png")
    except:
        pass

total_time = (time.time() - overall_start) / 3600
print(f"\n{'='*60}")
print(f"Complete! {len(all_results)}/{total} successful")
print(f"Total time: {total_time:.2f} hours")
print(f"{'='*60}")

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)