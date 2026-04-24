import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class TaxConfig:
    personal_allowance: float = 12570
    basic_rate_threshold: float = 50270
    higher_rate_threshold: float = 125140
    basic_rate: float = 0.20
    higher_rate: float = 0.40
    additional_rate: float = 0.45
    ni_threshold: float = 12570
    ni_rate_basic: float = 0.08
    ni_rate_higher: float = 0.02
    ni_upper_threshold: float = 50270


class Agent:
    def __init__(self, skill, preference=0.5):
        self.skill = skill
        self.alpha = preference
        self.labor_supply = 0.5

    def utility(self, consumption, labor):
        if consumption <= 0 or labor <= 0 or labor >= 1:
            return -1e10
        return (consumption ** self.alpha) * ((1 - labor) ** (1 - self.alpha))

    def optimize_labor(self, tax_function, epsilon=0.01, max_iter=50):
        best_labor = self.labor_supply
        best_utility = -np.inf

        # grid search
        for l in np.linspace(0.1, 0.95, 20):
            gross_income = self.skill * l
            net_income = gross_income - tax_function(gross_income)
            consumption = max(net_income, 100)
            u = self.utility(consumption, l)
            if u > best_utility:
                best_utility = u
                best_labor = l

        # refine
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
    def __init__(self, income_data_path=None, n_agents=2000, equity_weight=0.7):
        self.n_agents = n_agents
        self.equity_weight = equity_weight
        self.epsilon_swf = 1.2
        self.income_distribution = self._load_ons_data(income_data_path)
        self.agents = self._create_agents()
        self.tax_config = TaxConfig()
        self.history = {"gini": [], "revenue": [], "labor_supply": [], "utility": [], "reward": []}

    def _load_ons_data(self, path):
        # ONS data from Table 13
        income_by_decile = [14439, 24278, 32752, 40812, 50952, 60758, 72331, 85780, 108199, 198359]
        return pd.DataFrame({"decile": range(1, 11), "mean_income": income_by_decile})

    def _create_agents(self):
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

    def _uk_2024_tax_function(self, gross_income):
        config = self.tax_config
        tax = 0.0

        # personal allowance taper
        if gross_income <= 100000:
            pa = config.personal_allowance
        elif gross_income >= 125140:
            pa = 0.0
        else:
            reduction = (gross_income - 100000) / 2
            pa = max(0.0, config.personal_allowance - reduction)

        # income tax
        if gross_income > pa:
            taxable = gross_income - pa
            basic_band = config.basic_rate_threshold - pa
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

        # NI
        if gross_income > config.ni_threshold:
            ni_taxable = gross_income - config.ni_threshold
            basic_ni = config.ni_upper_threshold - config.ni_threshold
            if ni_taxable <= basic_ni:
                tax += ni_taxable * config.ni_rate_basic
            else:
                tax += basic_ni * config.ni_rate_basic
                tax += (ni_taxable - basic_ni) * config.ni_rate_higher

        return tax

    def _rl_tax_function(self, gross_income, policy_params):
        allowance = np.clip(policy_params[0], 12000, 20000)
        bracket1 = np.clip(policy_params[1], allowance + 1000, 100000)
        bracket2 = np.clip(policy_params[2], bracket1 + 1000, 200000)
        rate1 = np.clip(policy_params[3], 0.10, 0.25)
        rate2 = np.clip(policy_params[4], 0.25, 0.45)
        rate3 = np.clip(policy_params[5], 0.40, 0.60)

        # enforce progressive rates
        rate2 = max(rate2, rate1 + 0.05)
        rate3 = max(rate3, rate2 + 0.05)

        tax = 0.0

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

        # keep NI same as baseline
        config = self.tax_config
        if gross_income > config.ni_threshold:
            ni_taxable = gross_income - config.ni_threshold
            basic_ni = config.ni_upper_threshold - config.ni_threshold
            if ni_taxable <= basic_ni:
                tax += ni_taxable * config.ni_rate_basic
            else:
                tax += basic_ni * config.ni_rate_basic
                tax += (ni_taxable - basic_ni) * config.ni_rate_higher

        return tax

    def calculate_gini(self, incomes):
        sorted_incomes = np.sort(incomes)
        n = len(sorted_incomes)
        if n == 0 or np.sum(sorted_incomes) <= 0:
            return 0.0
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_incomes)) / (n * np.sum(sorted_incomes)) - (n + 1) / n
        return gini

    def calculate_atkinson_swf(self, net_incomes, epsilon=1.2):
        incomes = np.maximum(net_incomes, 100)
        if epsilon == 1.0:
            return np.exp(np.mean(np.log(incomes)))
        n = len(incomes)
        sum_term = np.sum(incomes ** (1 - epsilon)) / n
        return sum_term ** (1 / (1 - epsilon))

    def simulate(self, policy_params=None, use_uk_system=False):
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

        revenue = np.sum(gross_incomes - net_incomes)
        labor = np.sum([agent.labor_supply for agent in self.agents])
        gini = self.calculate_gini(net_incomes)
        avg_utility = np.mean(utilities)
        welfare = self.calculate_atkinson_swf(net_incomes, epsilon=self.epsilon_swf)

        return {
            "gini": gini,
            "revenue": revenue,
            "labor_supply": labor,
            "avg_utility": avg_utility,
            "social_welfare": welfare,
            "gross_incomes": gross_incomes,
            "net_incomes": net_incomes,
        }

    def get_reward(self, policy_params):
        results = self.simulate(policy_params)
        reward = results["social_welfare"]

        if not hasattr(self, "_uk_revenue_target"):
            uk_results = self.benchmark_uk_system()
            self._uk_revenue_target = uk_results["revenue"] * 0.90

        revenue_ratio = results["revenue"] / self._uk_revenue_target

        if revenue_ratio < 1.0:
            reward += -1000 * (1.0 - revenue_ratio)
        else:
            reward += min(5, 5 * (revenue_ratio - 1.0))

        return reward

    def benchmark_uk_system(self):
        return self.simulate(use_uk_system=True)


if __name__ == "__main__":
    print("Testing tax environment...")
    env = TaxEnvironment(n_agents=1000, equity_weight=0.7)
    
    print(f"\nAgents: {len(env.agents):,}")
    print(f"Epsilon: {env.epsilon_swf}")
    
    print("\nUK 2024/25 System:")
    uk = env.benchmark_uk_system()
    print(f"Gini: {uk['gini']:.4f}")
    print(f"Revenue: £{uk['revenue']:,.0f}")
    print(f"Welfare: {uk['social_welfare']:.2f}")
    
    print("\nTest policy:")
    test = np.array([12000, 35000, 80000, 0.15, 0.35, 0.50])
    opt = env.simulate(test)
    print(f"Gini: {opt['gini']:.4f}")
    print(f"Revenue: £{opt['revenue']:,.0f}")
    print(f"Welfare: {opt['social_welfare']:.2f}")