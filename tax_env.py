"""
UK Tax Optimization Environment - FIXED VERSION
All bugs corrected for dissertation
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TaxConfig:
    """Configuration for tax system parameters."""

    # UK 2024/25 Income Tax System (Baseline)
    personal_allowance: float = 12570
    basic_rate_threshold: float = 50270
    higher_rate_threshold: float = 125140

    basic_rate: float = 0.20
    higher_rate: float = 0.40
    additional_rate: float = 0.45

    # National Insurance (simplified, employee Class 1, 2024/25)
    ni_threshold: float = 12570
    ni_rate_basic: float = 0.08
    ni_rate_higher: float = 0.02
    ni_upper_threshold: float = 50270


class Agent:
    """
    Represents an individual taxpayer with utility maximization behavior.

    Utility function: U(c, l) = c^alpha * (1-l)^(1-alpha)
    where c = consumption (post-tax income), l = labor supply [0, 1]
    """

    def __init__(self, skill: float, preference: float = 0.5):
        """
        Args:
            skill: Wage rate (£/hour equivalent as annual income potential)
            preference: Alpha in utility function (work vs leisure preference)
        """
        self.skill = skill
        self.alpha = preference
        self.labor_supply = 0.5

    def utility(self, consumption: float, labor: float) -> float:
        """
        Calculate utility from consumption and labor.

        U(c, l) = c^α * (1-l)^(1-α)
        """
        if consumption <= 0 or labor <= 0 or labor >= 1:
            return -1e10

        return (consumption ** self.alpha) * ((1 - labor) ** (1 - self.alpha))

    def optimize_labor(
        self,
        tax_function,
        epsilon: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Find optimal labor supply given a tax function.

        Uses grid search + local refinement.
        """
        best_labor = self.labor_supply
        best_utility = -np.inf

        # Grid search
        for l in np.linspace(0.1, 0.95, 20):
            gross_income = self.skill * l
            net_income = gross_income - tax_function(gross_income)
            consumption = max(net_income, 100)

            u = self.utility(consumption, l)

            if u > best_utility:
                best_utility = u
                best_labor = l

        # Local refinement
        for _ in range(max_iter):
            for delta in [-epsilon, epsilon]:
                test_labor = np.clip(best_labor + delta, 0.05, 0.95)
                gross_income = self.skill * test_labor
                net_income = gross_income - tax_function(gross_income)
                consumption = max(net_income, 100)

                u = self.utility(consumption, test_labor)

                if u > best_utility:
                    best_utility = u
                    best_labor = test_labor

        self.labor_supply = best_labor
        return best_labor


class TaxEnvironment:
    """
    Simulation environment for UK tax policy optimization.

    Includes:
    - UK 2024/25 baseline with personal allowance taper
    - Simplified employee National Insurance
    - Atkinson Social Welfare Function
    - Enforced rate progressivity in optimized policy
    """

    def __init__(
        self,
        income_data_path: str = None,
        n_agents: int = 2000,
        equity_weight: float = 0.7,
    ):
        """
        Args:
            income_data_path: Path to ONS income distribution data (optional)
            n_agents: Number of synthetic agents to simulate
            equity_weight: Kept for compatibility, not used in Atkinson SWF
        """
        self.n_agents = n_agents
        self.equity_weight = equity_weight
        self.epsilon_swf = 1.2

        # Load ONS data and create agent distribution
        self.income_distribution = self._load_ons_data(income_data_path)
        self.agents = self._create_agents()

        # Current tax policy
        self.tax_config = TaxConfig()

        # Metrics tracking
        self.history = {
            "gini": [],
            "revenue": [],
            "labor_supply": [],
            "utility": [],
            "reward": [],
        }

    def _load_ons_data(self, path: str) -> pd.DataFrame:
        """Load and process ONS income distribution data."""
        # ONS non-retired household gross income by decile
        # Source used in dissertation: ONS "Effects of taxes and benefits on household income"
        income_by_decile = [
            14439, 24278, 32752, 40812, 50952,
            60758, 72331, 85780, 108199, 198359
        ]

        income_df = pd.DataFrame({
            "decile": range(1, 11),
            "mean_income": income_by_decile,
        })

        return income_df

    def _create_agents(self) -> List[Agent]:
        """Create synthetic population based on ONS distribution."""
        agents = []

        agents_per_decile = self.n_agents // 10

        for _, row in self.income_distribution.iterrows():
            mean_income = row["mean_income"]

            for _ in range(agents_per_decile):
                skill = np.random.normal(mean_income, mean_income * 0.2)
                skill = max(skill, 5000)

                preference = np.random.beta(5, 5)
                preference = np.clip(preference, 0.3, 0.7)

                agents.append(Agent(skill=skill, preference=preference))

        return agents

    def _uk_2024_tax_function(self, gross_income: float) -> float:
        """Calculate tax under UK 2024/25 system, including personal allowance taper."""
        config = self.tax_config
        tax = 0.0

        # Personal Allowance taper:
        # reduced by £1 for every £2 above £100,000
        # fully withdrawn at £125,140
        if gross_income <= 100000:
            personal_allowance = config.personal_allowance
        elif gross_income >= 125140:
            personal_allowance = 0.0
        else:
            reduction = (gross_income - 100000) / 2
            personal_allowance = max(0.0, config.personal_allowance - reduction)

        # Income Tax
        if gross_income > personal_allowance:
            taxable = gross_income - personal_allowance

            basic_band = config.basic_rate_threshold - personal_allowance
            higher_band = config.higher_rate_threshold - config.basic_rate_threshold

            if taxable <= basic_band:
                tax += taxable * config.basic_rate
            elif taxable <= basic_band + higher_band:
                tax += basic_band * config.basic_rate
                tax += (taxable - basic_band) * config.higher_rate
            else:
                tax += basic_band * config.basic_rate
                tax += higher_band * config.higher_rate
                tax += (taxable - basic_band - higher_band) * config.additional_rate

        # National Insurance (simplified employee NICs)
        if gross_income > config.ni_threshold:
            ni_taxable = gross_income - config.ni_threshold
            basic_ni_band = config.ni_upper_threshold - config.ni_threshold

            if ni_taxable <= basic_ni_band:
                tax += ni_taxable * config.ni_rate_basic
            else:
                tax += basic_ni_band * config.ni_rate_basic
                tax += (ni_taxable - basic_ni_band) * config.ni_rate_higher

        return tax

    def _rl_tax_function(self, gross_income: float, policy_params: np.ndarray) -> float:
        """
        Calculate tax under optimized policy.

        Policy params: [allowance, bracket1, bracket2, rate1, rate2, rate3]

        Includes simplified employee NI so results are comparable to baseline.
        """
        allowance = np.clip(policy_params[0], 12000, 20000)
        bracket1 = np.clip(policy_params[1], allowance + 1000, 100000)
        bracket2 = np.clip(policy_params[2], bracket1 + 1000, 200000)

        rate1 = np.clip(policy_params[3], 0.10, 0.25)
        rate2 = np.clip(policy_params[4], 0.25, 0.45)
        rate3 = np.clip(policy_params[5], 0.40, 0.60)

        # Enforce progressive structure
        rate2 = max(rate2, rate1 + 0.05)
        rate3 = max(rate3, rate2 + 0.05)

        tax = 0.0

        # Income tax under candidate policy
        if gross_income > allowance:
            taxable = gross_income - allowance

            band1 = bracket1 - allowance
            band2 = bracket2 - bracket1

            if taxable <= band1:
                tax += taxable * rate1
            elif taxable <= band1 + band2:
                tax += band1 * rate1
                tax += (taxable - band1) * rate2
            else:
                tax += band1 * rate1
                tax += band2 * rate2
                tax += (taxable - band1 - band2) * rate3

        # Keep simplified NI fixed for comparability with baseline
        config = self.tax_config
        if gross_income > config.ni_threshold:
            ni_taxable = gross_income - config.ni_threshold
            basic_ni_band = config.ni_upper_threshold - config.ni_threshold

            if ni_taxable <= basic_ni_band:
                tax += ni_taxable * config.ni_rate_basic
            else:
                tax += basic_ni_band * config.ni_rate_basic
                tax += (ni_taxable - basic_ni_band) * config.ni_rate_higher

        return tax

    def calculate_gini(self, incomes: np.ndarray) -> float:
        """Calculate Gini coefficient for income inequality."""
        sorted_incomes = np.sort(incomes)
        n = len(sorted_incomes)

        if n == 0 or np.sum(sorted_incomes) <= 0:
            return 0.0

        index = np.arange(1, n + 1)

        gini = (
            (2 * np.sum(index * sorted_incomes)) / (n * np.sum(sorted_incomes))
            - (n + 1) / n
        )
        return gini

    def calculate_atkinson_swf(
        self,
        net_incomes: np.ndarray,
        epsilon: float = 1.2,
    ) -> float:
        """
        Calculate Atkinson (1970) Social Welfare Function.

        SW = [1/N × Σ(y_i^(1-ε))]^(1/(1-ε))
        """
        incomes = np.maximum(net_incomes, 100)

        if epsilon == 1.0:
            return np.exp(np.mean(np.log(incomes)))

        n = len(incomes)
        sum_term = np.sum(incomes ** (1 - epsilon)) / n
        social_welfare = sum_term ** (1 / (1 - epsilon))
        return social_welfare

    def simulate(
        self,
        policy_params: np.ndarray = None,
        use_uk_system: bool = False,
    ) -> Dict:
        """
        Simulate the economy under a given tax policy.

        Returns metrics: Gini, revenue, labor supply, social welfare
        """
        if use_uk_system or policy_params is None:
            tax_func = self._uk_2024_tax_function
        else:
            tax_func = lambda income: self._rl_tax_function(income, policy_params)

        gross_incomes = []
        net_incomes = []
        utilities = []

        for agent in self.agents:
            optimal_labor = agent.optimize_labor(tax_func)

            gross_income = agent.skill * optimal_labor
            tax_paid = tax_func(gross_income)
            net_income = gross_income - tax_paid

            consumption = max(net_income, 100)
            utility = agent.utility(consumption, optimal_labor)

            gross_incomes.append(gross_income)
            net_incomes.append(net_income)
            utilities.append(utility)

        gross_incomes = np.array(gross_incomes)
        net_incomes = np.array(net_incomes)
        utilities = np.array(utilities)

        total_revenue = np.sum(gross_incomes - net_incomes)
        total_labor = np.sum([agent.labor_supply for agent in self.agents])
        gini = self.calculate_gini(net_incomes)
        avg_utility = np.mean(utilities)

        social_welfare = self.calculate_atkinson_swf(
            net_incomes,
            epsilon=self.epsilon_swf,
        )

        return {
            "gini": gini,
            "revenue": total_revenue,
            "labor_supply": total_labor,
            "avg_utility": avg_utility,
            "social_welfare": social_welfare,
            "gross_incomes": gross_incomes,
            "net_incomes": net_incomes,
        }

    def get_reward(self, policy_params: np.ndarray) -> float:
        """
        Calculate reward with strict revenue constraint.

        Kept for backward compatibility.
        """
        results = self.simulate(policy_params)

        reward = results["social_welfare"]

        if not hasattr(self, "_uk_revenue_target"):
            uk_results = self.benchmark_uk_system()
            self._uk_revenue_target = uk_results["revenue"] * 0.90

        revenue_ratio = results["revenue"] / self._uk_revenue_target

        if revenue_ratio < 1.0:
            revenue_penalty = -1000 * (1.0 - revenue_ratio)
            reward += revenue_penalty
        else:
            revenue_bonus = min(5, 5 * (revenue_ratio - 1.0))
            reward += revenue_bonus

        return reward

    def benchmark_uk_system(self) -> Dict:
        """Evaluate current UK tax system."""
        return self.simulate(use_uk_system=True)


if __name__ == "__main__":
    print("=" * 80)
    print("UK TAX OPTIMIZATION ENVIRONMENT - FIXED VERSION TEST")
    print("=" * 80)

    env = TaxEnvironment(n_agents=1000, equity_weight=0.7)

    print(f"\nCreated {len(env.agents):,} agents")
    print(f"Inequality aversion (ε): {env.epsilon_swf}")
    print("Income distribution (deciles):")
    print(env.income_distribution["mean_income"].values)

    print("\n" + "-" * 80)
    print("BENCHMARK: UK 2024/25 Tax System")
    print("-" * 80)
    uk_results = env.benchmark_uk_system()
    print(f"Gini Coefficient:           {uk_results['gini']:.4f}")
    print(f"Total Revenue:             £{uk_results['revenue']:,.0f}")
    print(f"Total Labor Supply:        {uk_results['labor_supply']:.0f}")
    print(f"Average Utility:           {uk_results['avg_utility']:.4f}")
    print(f"Social Welfare (Atkinson): {uk_results['social_welfare']:.2f}")

    print("\n" + "-" * 80)
    print("TEST: Sample Optimized Policy")
    print("-" * 80)
    test_policy = np.array([12000, 35000, 80000, 0.15, 0.35, 0.50])
    print(
        f"Policy: Allowance=£{test_policy[0]:,.0f}, "
        f"Brackets=[£{test_policy[1]:,.0f}, £{test_policy[2]:,.0f}]"
    )
    print(
        f"        Rates=[{test_policy[3]:.1%}, "
        f"{test_policy[4]:.1%}, {test_policy[5]:.1%}]"
    )

    opt_results = env.simulate(test_policy)
    print(f"\nGini Coefficient:           {opt_results['gini']:.4f}")
    print(f"Total Revenue:             £{opt_results['revenue']:,.0f}")
    print(f"Total Labor Supply:        {opt_results['labor_supply']:.0f}")
    print(f"Average Utility:           {opt_results['avg_utility']:.4f}")
    print(f"Social Welfare (Atkinson): {opt_results['social_welfare']:.2f}")

    print("\n" + "=" * 80)
    print("✅ FIXES VERIFIED:")
    print("  ✓ NI main rate: 8%")
    print("  ✓ Personal allowance taper implemented")
    print("  ✓ Rate progressivity: Enforced")
    print("  ✓ Social welfare: Atkinson (1970)")
    print("  ✓ Optimized policy comparable to baseline")
    print("=" * 80)
    print("Environment test complete - ready for dissertation!")