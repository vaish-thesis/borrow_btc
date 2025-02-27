import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import base64
import io

# Set page config
st.set_page_config(
    page_title="Bitcoin Housing Strategy Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BTCHousingModel:
    """
    Model for comparing 3 scenarios of using Bitcoin for house purchases:
    A) Sell BTC to buy house
    B) Borrow against BTC to buy house
    C) Use BTC as secondary collateral
    
    This model uses Geometric Brownian Motion (GBM) to simulate Bitcoin price paths.
    """
    
    def __init__(self, 
                 initial_btc,              # B₀: Initial BTC holdings
                 initial_btc_price,        # P₀: BTC price at time t=0
                 btc_basis_price,          # Price at which BTC was acquired (for tax purposes)
                 btc_appreciation_rate,    # μ: BTC annual expected return (drift)
                 initial_house_price,      # H₀: Initial house price (USD)
                 house_appreciation_rate,  # rₕ: Rate of house appreciation
                 capital_gains_tax,        # τ: Capital gains tax rate
                 loan_to_value_ratio,      # LTV: Loan to value ratio
                 mortgage_rate,            # iₛₜ: Mortgage rate
                 time_horizon,             # T: Time horizon (years)
                 btc_volatility,           # σ: BTC volatility (annual)
                 inflation_rate,           # π: Inflation rate
                 btc_selling_fee,          # F_BTC: Fee for selling BTC (%)
                 house_purchase_fee,       # F_House: House purchase fee (%)
                 annual_house_cost,        # f_House: Annual recurring house cost (%)
                 num_simulations=500,      # Number of Monte Carlo simulations
                 time_steps_per_year=12    # Time steps per year (monthly)
                ):
        
        # Store parameters
        self.B0 = initial_btc
        self.P0 = initial_btc_price
        self.P_basis = btc_basis_price
        self.mu = btc_appreciation_rate    # Using μ as drift in GBM
        self.H0 = initial_house_price
        self.rh = house_appreciation_rate
        self.tau = capital_gains_tax
        self.LTV = loan_to_value_ratio
        self.i_st = mortgage_rate
        self.T = time_horizon
        self.sigma = btc_volatility        # Using σ directly for GBM
        self.pi = inflation_rate
        self.F_BTC = btc_selling_fee
        self.F_House = house_purchase_fee
        self.f_House = annual_house_cost
        self.num_simulations = num_simulations
        self.time_steps_per_year = time_steps_per_year
        
        # Derived values
        self.initial_btc_value = self.B0 * self.P0
        self.total_steps = int(self.T * self.time_steps_per_year)
        self.dt = 1.0 / self.time_steps_per_year
        
        # Pre-generate all Bitcoin price paths for consistency across scenarios
        self.btc_price_paths = self._generate_btc_price_paths()
        
        # Pre-generate house price paths (less volatile but still stochastic)
        self.house_price_paths = self._generate_house_price_paths()
    
    def _generate_btc_price_paths(self):
        """
        Generate Bitcoin price paths using Geometric Brownian Motion (GBM)
        dS = μS dt + σS dW
        
        Returns a 2D numpy array of shape (num_simulations, total_steps+1)
        """
        # Initialize array for price paths
        price_paths = np.zeros((self.num_simulations, self.total_steps + 1))
        
        # Set initial price for all paths
        price_paths[:, 0] = self.P0
        
        # Generate random normal samples for Brownian motion
        # Using antithetic variates for variance reduction
        half_sims = self.num_simulations // 2
        Z = np.random.normal(0, 1, (half_sims, self.total_steps))
        Z_antithetic = -Z  # Antithetic pairs
        Z = np.vstack((Z, Z_antithetic))
        
        # Ensure we have exactly num_simulations paths
        if Z.shape[0] < self.num_simulations:
            extra = np.random.normal(0, 1, (self.num_simulations - Z.shape[0], self.total_steps))
            Z = np.vstack((Z, extra))
        
        # Drift and volatility terms for GBM
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        # Generate price paths
        for t in range(1, self.total_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * Z[:, (t-1) % Z.shape[1]])
        
        return price_paths
    
    def _generate_house_price_paths(self):
        """
        Generate house price paths using GBM with lower volatility
        
        Returns a 2D numpy array of shape (num_simulations, total_steps+1)
        """
        # House prices are less volatile than BTC
        house_volatility = self.rh * 0.3  # 30% of the appreciation rate
        
        # Initialize array for price paths
        price_paths = np.zeros((self.num_simulations, self.total_steps + 1))
        
        # Set initial price for all paths
        price_paths[:, 0] = self.H0
        
        # Generate random normal samples
        Z = np.random.normal(0, 1, (self.num_simulations, self.total_steps))
        
        # Drift and volatility terms
        drift = (self.rh - 0.5 * house_volatility**2) * self.dt
        diffusion = house_volatility * np.sqrt(self.dt)
        
        # Generate price paths
        for t in range(1, self.total_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
        return price_paths
    
    def _get_yearly_indices(self):
        """Return indices for yearly points in the simulation"""
        steps_per_year = self.time_steps_per_year
        return [i * steps_per_year for i in range(self.T + 1)]
    
    def scenario_a_sell_btc(self):
        """
        Scenario A: Selling BTC to buy a House
        
        Run Monte Carlo simulations and return aggregated results
        """
        # Array to store results from each simulation
        net_values = np.zeros(self.num_simulations)
        btc_values_if_held = np.zeros(self.num_simulations)
        
        # Run simulations
        for sim in range(self.num_simulations):
            # Extract this simulation's price paths
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            
            # 1. Selling BTC at time 0
            initial_btc_value = self.B0 * self.P0  # Same for all simulations
            btc_selling_fee = initial_btc_value * self.F_BTC
            capital_gains = self.B0 * (self.P0 - self.P_basis)
            capital_gains_tax = capital_gains * self.tau
            net_proceeds = initial_btc_value - btc_selling_fee - capital_gains_tax
            
            # 2. Buy House
            house_cost = self.H0 * (1 + self.F_House)
            
            # Check if enough money to buy house
            can_afford = net_proceeds >= house_cost
            remaining_cash = net_proceeds - house_cost if can_afford else 0
            
            if not can_afford:
                # If can't afford house, just hold BTC instead
                net_values[sim] = self.B0 * btc_prices[-1]
                btc_values_if_held[sim] = self.B0 * btc_prices[-1]
                continue
            
            # 3. Holding cost over time (Present Value)
            yearly_indices = self._get_yearly_indices()
            holding_costs_pv = 0
            
            for i, idx in enumerate(yearly_indices):
                if i == 0:  # Skip the initial year
                    continue
                house_value_t = house_prices[idx]
                yearly_cost = self.f_House * house_value_t
                # Discount to present value
                holding_costs_pv += yearly_cost / ((1 + self.pi) ** i)
            
            # 4. Final values at time T
            final_house_value = house_prices[-1]
            final_btc_value_if_held = self.B0 * btc_prices[-1]
            
            # 5. Net Gain
            net_value = final_house_value - house_cost - holding_costs_pv + remaining_cash
            
            # Store results
            net_values[sim] = net_value
            btc_values_if_held[sim] = final_btc_value_if_held
        
        # Compute percentiles and metrics
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        btc_held_percentiles = np.percentile(btc_values_if_held, percentiles)
        
        # Get median value for detailed output
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        median_house_value = self.house_price_paths[median_sim_idx, -1]
        
        # Opportunity cost - what if we kept the BTC instead?
        opportunity_cost = btc_values_if_held[median_sim_idx] - initial_btc_value
        
        return {
            "scenario": "A - Sell BTC to Buy House",
            "can_afford_house": net_proceeds >= house_cost,
            "initial_btc_value": initial_btc_value,
            "net_proceeds_after_tax": net_proceeds,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "holding_costs_pv": holding_costs_pv,  # Using median sim for this
            "final_house_value": median_house_value,
            "final_btc_value_if_held": btc_values_if_held[median_sim_idx],
            "net_value": net_values[median_sim_idx],
            "opportunity_cost": opportunity_cost,
            "total_return": (net_values[median_sim_idx] / house_cost) * 100 if house_cost > 0 else 0,
            "btc_return_if_held": ((btc_values_if_held[median_sim_idx] / initial_btc_value) - 1) * 100,
            # Distribution stats
            "net_value_percentiles": dict(zip(
                ["p10", "p25", "p50", "p75", "p90"],
                net_value_percentiles
            )),
            "btc_held_percentiles": dict(zip(
                ["p10", "p25", "p50", "p75", "p90"],
                btc_held_percentiles
            )),
            "all_net_values": net_values.tolist(),
            "all_btc_values": btc_values_if_held.tolist()
        }
    
    def scenario_b_borrow_against_btc(self):
        """
        Scenario B: Borrowing Against BTC to buy a house
        
        Run Monte Carlo simulations and return aggregated results
        """
        # Arrays to store results
        net_values = np.zeros(self.num_simulations)
        liquidation_occurred = np.zeros(self.num_simulations, dtype=bool)
        liquidation_times = np.full(self.num_simulations, np.nan)
        
        # Initial loan parameters (same for all simulations)
        loan_amount = self.LTV * self.B0 * self.P0
        house_cost = self.H0 * (1 + self.F_House)
        can_afford = loan_amount >= house_cost
        remaining_cash = loan_amount - house_cost if can_afford else 0
        
        # Skip simulation if loan doesn't cover house cost
        if not can_afford:
            return {
                "scenario": "B - Borrow Against BTC",
                "can_afford_house": False,
                "loan_amount": loan_amount,
                "house_cost": house_cost,
                "net_value": 0,
                "total_return": 0
            }
        
        # Liquidation threshold (typically 125% of LTV)
        liquidation_threshold = 1.25 * self.LTV
        
        # Run simulations
        for sim in range(self.num_simulations):
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            
            # Track loan value over time
            loan_values = np.zeros(self.total_steps + 1)
            loan_values[0] = loan_amount
            
            # Compute loan growth with compound interest
            for t in range(1, self.total_steps + 1):
                loan_values[t] = loan_values[t-1] * (1 + self.i_st/self.time_steps_per_year)
            
            # Check for liquidation at each time step
            btc_values = self.B0 * btc_prices
            current_ltv = loan_values / btc_values
            
            # Find first liquidation point if any
            liquidation_points = np.where(current_ltv > liquidation_threshold)[0]
            
            if len(liquidation_points) > 0:
                liquidation_occurred[sim] = True
                first_liquidation = liquidation_points[0]
                liquidation_times[sim] = first_liquidation / self.time_steps_per_year
                
                # In case of liquidation, we lose the BTC but keep the house
                final_btc_value = 0
            else:
                # No liquidation
                final_btc_value = btc_values[-1]
            
            # Final values
            final_house_value = house_prices[-1]
            final_loan_value = loan_values[-1]
            
            # Net value calculation
            net_values[sim] = final_house_value + final_btc_value - final_loan_value + remaining_cash
        
        # Compute statistics
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        
        # Get median simulation for detailed output
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        median_btc_prices = self.btc_price_paths[median_sim_idx]
        median_btc_values = self.B0 * median_btc_prices
        
        median_loan_values = np.zeros(self.total_steps + 1)
        median_loan_values[0] = loan_amount
        for t in range(1, self.total_steps + 1):
            median_loan_values[t] = median_loan_values[t-1] * (1 + self.i_st/self.time_steps_per_year)
        
        median_ltv = (median_loan_values / median_btc_values) * 100  # Convert to percentage
        
        # Time points for charts (in years)
        time_points = np.linspace(0, self.T, self.total_steps + 1)
        
        # Probability of liquidation
        liquidation_probability = np.mean(liquidation_occurred) * 100
        
        return {
            "scenario": "B - Borrow Against BTC",
            "can_afford_house": can_afford,
            "loan_amount": loan_amount,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "loan_with_interest": median_loan_values[-1],
            "final_btc_price": median_btc_prices[-1],
            "final_btc_value": median_btc_values[-1] if not liquidation_occurred[median_sim_idx] else 0,
            "final_house_value": self.house_price_paths[median_sim_idx, -1],
            "liquidation_occurred": liquidation_occurred[median_sim_idx],
            "liquidation_time": liquidation_times[median_sim_idx] if liquidation_occurred[median_sim_idx] else None,
            "net_value": net_values[median_sim_idx],
            "total_return": (net_values[median_sim_idx] / self.initial_btc_value) * 100,
            "time_points": time_points.tolist(),
            "btc_value_history": median_btc_values.tolist(),
            "loan_value_history": median_loan_values.tolist(),
            "ltv_history": median_ltv.tolist(),
            # Additional statistics
            "liquidation_probability": liquidation_probability,
            "net_value_percentiles": dict(zip(
                ["p10", "p25", "p50", "p75", "p90"],
                net_value_percentiles
            )),
            "all_net_values": net_values.tolist()
        }
    
    def scenario_c_btc_collateral(self):
        """
        Scenario C: BTC as Secondary Collateral (Self-paying Mortgage)
        
        Run Monte Carlo simulations and return aggregated results
        """
        # Arrays to store results
        net_values = np.zeros(self.num_simulations)
        liquidation_occurred = np.zeros(self.num_simulations, dtype=bool)
        liquidation_times = np.full(self.num_simulations, np.nan)
        debt_paid_off = np.zeros(self.num_simulations, dtype=bool)
        
        # Initial loan parameters (same for all simulations)
        loan_amount = self.H0 * (1 + self.F_House)
        
        # For tracking median simulation
        median_debt_history = None
        median_house_value_history = None
        median_btc_value_history = None
        median_collateral_ratio_history = None
        
        # Get yearly indices for tracking
        yearly_indices = self._get_yearly_indices()
        
        # Run simulations
        for sim in range(self.num_simulations):
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            
            # Initialize tracking arrays for this simulation
            debt_history = np.zeros(len(yearly_indices))
            house_value_history = np.zeros(len(yearly_indices))
            btc_value_history = np.zeros(len(yearly_indices))
            collateral_ratio_history = np.zeros(len(yearly_indices))
            
            # Initial values
            debt_remaining = loan_amount
            debt_history[0] = debt_remaining
            house_value_history[0] = house_prices[0]
            btc_value_history[0] = self.B0 * btc_prices[0]
            collateral_ratio_history[0] = (house_value_history[0] + btc_value_history[0]) / debt_remaining
            
            # Track liquidation
            sim_liquidation_occurred = False
            sim_liquidation_time = None
            
            # Simulate over yearly steps
            for t in range(1, len(yearly_indices)):
                idx = yearly_indices[t]
                
                # Update house and BTC values
                house_value = house_prices[idx]
                house_value_history[t] = house_value
                
                btc_value = self.B0 * btc_prices[idx]
                btc_value_history[t] = btc_value
                
                # Calculate debt with interest
                debt_with_interest = debt_remaining * (1 + self.i_st)
                
                # Check if BTC + house value can cover the debt
                total_collateral = house_value + btc_value
                required_collateral = debt_with_interest * 1.25  # 125% LTV requirement
                
                if total_collateral < required_collateral and not sim_liquidation_occurred:
                    sim_liquidation_occurred = True
                    sim_liquidation_time = t
                    # In case of liquidation, we lose the BTC
                    btc_value = 0
                    btc_value_history[t:] = 0
                    excess_value = 0
                else:
                    # Calculate excess collateral that can be used to pay down principal
                    excess_value = max(0, total_collateral - required_collateral)
                
                # Apply excess to debt reduction
                principal_reduction = min(excess_value, debt_with_interest)
                debt_remaining = debt_with_interest - principal_reduction
                debt_history[t] = debt_remaining
                
                # Calculate collateral ratio
                if debt_remaining > 0:
                    collateral_ratio = (house_value + btc_value) / debt_remaining
                else:
                    collateral_ratio = float('inf')  # No debt
                    debt_paid_off[sim] = True
                
                collateral_ratio_history[t] = min(collateral_ratio, 10)  # Cap for visualization
                
                # If debt is fully paid, fill the rest with zeros
                if debt_remaining <= 0:
                    debt_remaining = 0
                    debt_history[t:] = 0
                    debt_paid_off[sim] = True
                    break
            
            # Store liquidation results
            liquidation_occurred[sim] = sim_liquidation_occurred
            if sim_liquidation_time is not None:
                liquidation_times[sim] = sim_liquidation_time
            
            # Calculate net value
            final_house_value = house_prices[-1]
            final_btc_value = self.B0 * btc_prices[-1] if not sim_liquidation_occurred else 0
            final_debt = debt_remaining
            
            net_values[sim] = final_house_value + final_btc_value - final_debt
        
        # Compute statistics
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        
        # Get median simulation for detailed output
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        
        # Re-run the median simulation to get detailed history
        btc_prices = self.btc_price_paths[median_sim_idx]
        house_prices = self.house_price_paths[median_sim_idx]
        
        # Initialize tracking arrays for median simulation
        debt_history = np.zeros(len(yearly_indices))
        house_value_history = np.zeros(len(yearly_indices))
        btc_value_history = np.zeros(len(yearly_indices))
        collateral_ratio_history = np.zeros(len(yearly_indices))
        
        # Initial values
        debt_remaining = loan_amount
        debt_history[0] = debt_remaining
        house_value_history[0] = house_prices[0]
        btc_value_history[0] = self.B0 * btc_prices[0]
        collateral_ratio_history[0] = (house_value_history[0] + btc_value_history[0]) / debt_remaining
        
        # Track liquidation
        median_liquidation_occurred = False
        median_liquidation_time = None
        
        # Simulate over yearly steps for median simulation
        for t in range(1, len(yearly_indices)):
            idx = yearly_indices[t]
            
            # Update house and BTC values
            house_value = house_prices[idx]
            house_value_history[t] = house_value
            
            btc_value = self.B0 * btc_prices[idx]
            btc_value_history[t] = btc_value
            
            # Calculate debt with interest
            debt_with_interest = debt_remaining * (1 + self.i_st)
            
            # Check if BTC + house value can cover the debt
            total_collateral = house_value + btc_value
            required_collateral = debt_with_interest * 1.25  # 125% LTV requirement
            
            if total_collateral < required_collateral and not median_liquidation_occurred:
                median_liquidation_occurred = True
                median_liquidation_time = t
                # In case of liquidation, we lose the BTC
                btc_value = 0
                btc_value_history[t:] = 0
                excess_value = 0
            else:
                # Calculate excess collateral that can be used to pay down principal
                excess_value = max(0, total_collateral - required_collateral)
            
            # Apply excess to debt reduction
            principal_reduction = min(excess_value, debt_with_interest)
            debt_remaining = debt_with_interest - principal_reduction
            debt_history[t] = debt_remaining
            
            # Calculate collateral ratio
            if debt_remaining > 0:
                collateral_ratio = (house_value + btc_value) / debt_remaining
            else:
                collateral_ratio = float('inf')  # No debt
            
            collateral_ratio_history[t] = min(collateral_ratio, 10)  # Cap for visualization
            
            # If debt is fully paid, fill the rest with zeros
            if debt_remaining <= 0:
                debt_remaining = 0
                debt_history[t:] = 0
                break
        
        # Time points for yearly charts
        time_points = np.arange(self.T + 1)
        
        # Probability of liquidation
        liquidation_probability = np.mean(liquidation_occurred) * 100
        
        # Probability of debt payoff
        debt_payoff_probability = np.mean(debt_paid_off) * 100
        
        return {
            "scenario": "C - BTC as Secondary Collateral",
            "initial_loan": loan_amount,
            "final_house_value": house_prices[-1],
            "final_btc_value": self.B0 * btc_prices[-1] if not median_liquidation_occurred else 0,
            "final_debt": debt_remaining,
            "liquidation_occurred": median_liquidation_occurred,
            "liquidation_time": median_liquidation_time,
            "debt_paid_off": debt_remaining == 0,
            "debt_history": debt_history.tolist(),
            "house_value_history": house_value_history.tolist(),
            "btc_value_history": btc_value_history.tolist(),
            "collateral_ratio_history": collateral_ratio_history.tolist(),
            "time_points": time_points.tolist(),
            "net_value": net_values[median_sim_idx],
            "total_return": (net_values[median_sim_idx] / self.initial_btc_value) * 100,
            # Additional statistics
            "liquidation_probability": liquidation_probability,
            "debt_payoff_probability": debt_payoff_probability,
            "net_value_percentiles": dict(zip(
                ["p10", "p25", "p50", "p75", "p90"],
                net_value_percentiles
            )),
            "all_net_values": net_values.tolist()
        }
    
    def simulate_scenarios(self):
        """
        Aggregate results from all three scenarios
        
        Since we now use Monte Carlo in the core functions, this just calls
        each scenario function and returns the results in the expected format.
        """
        # Run the three scenarios
        scenario_a = self.scenario_a_sell_btc()
        scenario_b = self.scenario_b_borrow_against_btc()
        scenario_c = self.scenario_c_btc_collateral()
        
        # Extract net value arrays
        scenario_a_results = np.array(scenario_a["all_net_values"])
        scenario_b_results = np.array(scenario_b["all_net_values"]) if "all_net_values" in scenario_b else np.zeros(self.num_simulations)
        scenario_c_results = np.array(scenario_c["all_net_values"])
        
        # Calculate percentiles for bull/bear/base cases
        percentiles = [10, 50, 90]  # 10% = bear, 50% = base, 90% = bull
        
        scenario_a_cases = np.percentile(scenario_a_results, percentiles)
        scenario_b_cases = np.percentile(scenario_b_results, percentiles)
        scenario_c_cases = np.percentile(scenario_c_results, percentiles)
        
        return {
            "Scenario A": {
                "Bear Case": scenario_a_cases[0],
                "Base Case": scenario_a_cases[1],
                "Bull Case": scenario_a_cases[2],
                "All Results": scenario_a_results.tolist()
            },
            "Scenario B": {
                "Bear Case": scenario_b_cases[0],
                "Base Case": scenario_b_cases[1],
                "Bull Case": scenario_b_cases[2],
                "All Results": scenario_b_results.tolist()
            },
            "Scenario C": {
                "Bear Case": scenario_c_cases[0],
                "Base Case": scenario_c_cases[1],
                "Bull Case": scenario_c_cases[2],
                "All Results": scenario_c_results.tolist()
            }
        }

def format_currency(value):
    """Format a value as USD currency"""
    return f"${value:,.2f}"

def format_percent(value):
    """Format a value as percentage"""
    return f"{value:.2f}%"

def generate_json_download_link(json_data, filename="btc_housing_inputs.json"):
    """Generate a download link for JSON data"""
    json_str = json.dumps(json_data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:text/json;base64,{b64}" download="{filename}">Download JSON</a>'
    return href

def run_analysis(inputs):
    """Run the analysis with the given inputs"""
    # Create model with inputs
    model = BTCHousingModel(
        initial_btc=inputs["initial_btc"],
        initial_btc_price=inputs["initial_btc_price"],
        btc_basis_price=inputs["btc_basis_price"],
        btc_appreciation_rate=inputs["btc_appreciation_rate"],
        initial_house_price=inputs["initial_house_price"],
        house_appreciation_rate=inputs["house_appreciation_rate"],
        capital_gains_tax=inputs["capital_gains_tax"],
        loan_to_value_ratio=inputs["loan_to_value_ratio"],
        mortgage_rate=inputs["mortgage_rate"],
        time_horizon=inputs["time_horizon"],
        btc_volatility=inputs["btc_volatility"],
        inflation_rate=inputs["inflation_rate"],
        btc_selling_fee=inputs["btc_selling_fee"],
        house_purchase_fee=inputs["house_purchase_fee"],
        annual_house_cost=inputs["annual_house_cost"],
        num_simulations=inputs.get("num_simulations", 500)
    )
    
    # Run scenarios - now fully based on Monte Carlo
    scenario_a = model.scenario_a_sell_btc()
    scenario_b = model.scenario_b_borrow_against_btc()
    scenario_c = model.scenario_c_btc_collateral()
    
    # Run simulations to get aggregated statistics
    simulations = model.simulate_scenarios()
    
    return model, scenario_a, scenario_b, scenario_c, simulations

def display_scenario_metrics(scenario, simulation_data, col):
    """Display metrics for a scenario in a Streamlit column"""
    col.subheader(scenario["scenario"])
    
    # Create a metrics section
    col.metric("Net Value", format_currency(scenario["net_value"]))
    col.metric("Total Return", format_percent(scenario["total_return"]))
    
    # Create an expander for more details
    with col.expander("Detailed Metrics"):
        if "can_afford_house" in scenario:
            st.write(f"**Can Afford House:** {'Yes' if scenario['can_afford_house'] else 'No'}")
        
        if "final_house_value" in scenario:
            st.write(f"**Final House Value:** {format_currency(scenario['final_house_value'])}")
        
        if scenario["scenario"].startswith("A"):
            st.write(f"**Final BTC Value (if held):** {format_currency(scenario['final_btc_value_if_held'])}")
            st.write(f"**Net Proceeds After Tax:** {format_currency(scenario['net_proceeds_after_tax'])}")
            st.write(f"**House Cost:** {format_currency(scenario['house_cost'])}")
            st.write(f"**Remaining Cash:** {format_currency(scenario['remaining_cash'])}")
            st.write(f"**Opportunity Cost:** {format_currency(scenario['opportunity_cost'])}")
        else:
            st.write(f"**Final BTC Value:** {format_currency(scenario['final_btc_value'])}")
        
        if "liquidation_occurred" in scenario and scenario["scenario"] != "A - Sell BTC to Buy House":
            st.write(f"**Liquidation Occurred:** {'Yes' if scenario['liquidation_occurred'] else 'No'}")
            if "liquidation_probability" in scenario:
                st.write(f"**Liquidation Probability:** {format_percent(scenario['liquidation_probability'])}")
            if scenario["liquidation_occurred"] and scenario["liquidation_time"] is not None:
                st.write(f"**Liquidation Time:** Year {scenario['liquidation_time']:.1f}")
        
        if "debt_paid_off" in scenario:
            st.write(f"**Debt Paid Off:** {'Yes' if scenario['debt_paid_off'] else 'No'}")
            if "debt_payoff_probability" in scenario:
                st.write(f"**Debt Payoff Probability:** {format_percent(scenario['debt_payoff_probability'])}")
        
        if scenario["scenario"].startswith("B"):
            st.write(f"**Loan Amount:** {format_currency(scenario['loan_amount'])}")
            st.write(f"**Loan with Interest:** {format_currency(scenario['loan_with_interest'])}")
        
        if scenario["scenario"].startswith("C"):
            st.write(f"**Initial Loan:** {format_currency(scenario['initial_loan'])}")
            st.write(f"**Final Debt:** {format_currency(scenario['final_debt'])}")
    
    # Display bull/bear/base cases
    with col.expander("Bull/Bear/Base Cases"):
        st.write(f"**Bear Case (10%):** {format_currency(simulation_data['Bear Case'])}")
        st.write(f"**Base Case (50%):** {format_currency(simulation_data['Base Case'])}")
        st.write(f"**Bull Case (90%):** {format_currency(simulation_data['Bull Case'])}")

def create_comparison_chart(scenario_a, scenario_b, scenario_c, simulations):
    """Create a chart comparing the three scenarios"""
    # Create a figure for the net value comparison
    fig = go.Figure()
    
    # Add bars for each scenario
    scenarios = ["Scenario A", "Scenario B", "Scenario C"]
    net_values = [
        scenario_a["net_value"],
        scenario_b["net_value"],
        scenario_c["net_value"]
    ]
    
    # Create bar chart
    fig.add_trace(go.Bar(
        x=scenarios,
        y=net_values,
        marker_color=['#3498db', '#2ecc71', '#e74c3c'],
        text=[format_currency(val) for val in net_values],
        textposition='auto',
        name='Net Value'
    ))
    
    # Update layout
    fig.update_layout(
        title='Net Value Comparison',
        xaxis_title='Scenario',
        yaxis_title='Net Value ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=500
    )
    
    return fig

def create_bull_bear_chart(simulations):
    """Create a chart comparing bull/bear/base cases"""
    # Data for the chart
    scenarios = ["Scenario A", "Scenario B", "Scenario C"]
    bear_cases = [
        simulations["Scenario A"]["Bear Case"],
        simulations["Scenario B"]["Bear Case"],
        simulations["Scenario C"]["Bear Case"]
    ]
    base_cases = [
        simulations["Scenario A"]["Base Case"],
        simulations["Scenario B"]["Base Case"],
        simulations["Scenario C"]["Base Case"]
    ]
    bull_cases = [
        simulations["Scenario A"]["Bull Case"],
        simulations["Scenario B"]["Bull Case"],
        simulations["Scenario C"]["Bull Case"]
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        x=scenarios,
        y=bear_cases,
        name='Bear Case',
        marker_color='#e74c3c',
        text=[format_currency(val) for val in bear_cases],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=base_cases,
        name='Base Case',
        marker_color='#3498db',
        text=[format_currency(val) for val in base_cases],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=bull_cases,
        name='Bull Case',
        marker_color='#2ecc71',
        text=[format_currency(val) for val in bull_cases],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Bull/Bear/Base Case Comparison',
        xaxis_title='Scenario',
        yaxis_title='Net Value ($)',
        yaxis=dict(tickformat='$,.0f'),
        barmode='group',
        height=500
    )
    
    return fig

def create_distribution_chart(simulations):
    """Create a histogram of simulation results"""
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Histogram(
        x=simulations["Scenario A"]["All Results"],
        name="Scenario A",
        opacity=0.75,
        marker_color='#3498db',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=simulations["Scenario B"]["All Results"],
        name="Scenario B",
        opacity=0.75,
        marker_color='#2ecc71',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=simulations["Scenario C"]["All Results"],
        name="Scenario C",
        opacity=0.75,
        marker_color='#e74c3c',
        nbinsx=30
    ))
    
    # Update layout
    fig.update_layout(
        title='Distribution of Simulation Results',
        xaxis_title='Net Value ($)',
        xaxis=dict(tickformat='$,.0f'),
        yaxis_title='Frequency',
        barmode='overlay',
        height=500
    )
    
    return fig

def create_scenario_charts(scenario_a, scenario_b, scenario_c):
    """Create charts specific to each scenario"""
    charts = {}
    
    # Scenario B: LTV over time
    if "time_points" in scenario_b and "ltv_history" in scenario_b:
        fig_ltv = go.Figure()
        
        fig_ltv.add_trace(go.Scatter(
            x=scenario_b["time_points"],
            y=scenario_b["ltv_history"],
            mode='lines',
            name='LTV Ratio',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add liquidation threshold line
        fig_ltv.add_trace(go.Scatter(
            x=scenario_b["time_points"],
            y=[125] * len(scenario_b["time_points"]),  # 125% LTV liquidation threshold
            mode='lines',
            name='Liquidation Threshold',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig_ltv.update_layout(
            title='Scenario B: Loan-to-Value Ratio Over Time',
            xaxis_title='Year',
            yaxis_title='LTV Ratio (%)',
            height=400
        )
        
        charts["ltv_chart"] = fig_ltv
    
    # Scenario C: Debt reduction over time
    if "time_points" in scenario_c and "debt_history" in scenario_c:
        fig_debt = go.Figure()
        
        fig_debt.add_trace(go.Scatter(
            x=scenario_c["time_points"],
            y=scenario_c["debt_history"],
            mode='lines',
            name='Debt Remaining',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig_debt.add_trace(go.Scatter(
            x=scenario_c["time_points"],
            y=scenario_c["house_value_history"],
            mode='lines',
            name='House Value',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig_debt.add_trace(go.Scatter(
            x=scenario_c["time_points"],
            y=scenario_c["btc_value_history"],
            mode='lines',
            name='BTC Value',
            line=dict(color='#3498db', width=2)
        ))
        
        fig_debt.update_layout(
            title='Scenario C: Asset Values and Debt Over Time',
            xaxis_title='Year',
            yaxis_title='Value ($)',
            yaxis=dict(tickformat='$,.0f'),
            height=400
        )
        
        charts["debt_chart"] = fig_debt
        
        # Collateral ratio chart
        if "collateral_ratio_history" in scenario_c:
            # Ratios already capped in the scenario function
            fig_collateral = go.Figure()
            
            fig_collateral.add_trace(go.Scatter(
                x=scenario_c["time_points"],
                y=scenario_c["collateral_ratio_history"],
                mode='lines',
                name='Collateral Ratio',
                line=dict(color='#9b59b6', width=2)
            ))
            
            # Add minimum required ratio line
            fig_collateral.add_trace(go.Scatter(
                x=scenario_c["time_points"],
                y=[1.25] * len(scenario_c["time_points"]),  # 125% required ratio
                mode='lines',
                name='Minimum Required Ratio',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig_collateral.update_layout(
                title='Scenario C: Collateral Ratio Over Time',
                xaxis_title='Year',
                yaxis_title='Collateral Ratio',
                height=400
            )
            
            charts["collateral_chart"] = fig_collateral
    
    return charts

def create_btc_price_path_chart(model):
    """Create a chart showing sample BTC price paths"""
    # Select a subset of price paths for visualization
    num_paths_to_show = 10
    path_indices = np.linspace(0, model.num_simulations-1, num_paths_to_show, dtype=int)
    
    fig = go.Figure()
    
    # Get time points in years
    time_points = np.linspace(0, model.T, model.total_steps + 1)
    
    # Add each path
    for i, idx in enumerate(path_indices):
        path = model.btc_price_paths[idx]
        
        # Use a color gradient from blue to red
        hue = 240 * (1 - i / (num_paths_to_show - 1))  # 240 (blue) to 0 (red)
        color = f"hsla({hue}, 100%, 50%, 0.5)"
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=path,
            mode='lines',
            name=f'Path {i+1}',
            line=dict(color=color, width=1),
            showlegend=(i == 0 or i == num_paths_to_show-1)
        ))
    
    # Add median path
    median_path_idx = np.argsort(model.btc_price_paths[:, -1])[model.num_simulations//2]
    median_path = model.btc_price_paths[median_path_idx]
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=median_path,
        mode='lines',
        name='Median Path',
        line=dict(color='black', width=2),
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title='Sample BTC Price Paths (GBM)',
        xaxis_title='Year',
        yaxis_title='BTC Price ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=400
    )
    
    return fig

def get_recommendation(scenario_a, scenario_b, scenario_c, simulations):
    """Get a recommendation based on the analysis"""
    # Determine best scenario based on base case net value
    scenarios = {
        "A": {"name": "Sell BTC to Buy House", "base_case": simulations["Scenario A"]["Base Case"]},
        "B": {"name": "Borrow Against BTC", "base_case": simulations["Scenario B"]["Base Case"]},
        "C": {"name": "BTC as Secondary Collateral", "base_case": simulations["Scenario C"]["Base Case"]}
    }
    
    best_scenario = max(scenarios.items(), key=lambda x: x[1]["base_case"])[0]
    
    # Calculate downside risk (base case - bear case)
    downside_risks = {}
    for scenario in ["A", "B", "C"]:
        base = simulations[f"Scenario {scenario}"]["Base Case"]
        bear = simulations[f"Scenario {scenario}"]["Bear Case"]
        downside_risks[scenario] = base - bear
    
    min_risk_scenario = min(downside_risks.items(), key=lambda x: x[1])[0]
    
    # Calculate upside potential (bull case - base case)
    upside_potentials = {}
    for scenario in ["A", "B", "C"]:
        bull = simulations[f"Scenario {scenario}"]["Bull Case"]
        base = simulations[f"Scenario {scenario}"]["Base Case"]
        upside_potentials[scenario] = bull - base
    
    max_upside_scenario = max(upside_potentials.items(), key=lambda x: x[1])[0]
    
    # Calculate probability metrics
    liquidation_prob_b = scenario_b.get("liquidation_probability", 0)
    liquidation_prob_c = scenario_c.get("liquidation_probability", 0)
    debt_payoff_prob_c = scenario_c.get("debt_payoff_probability", 0)
    
    # Generate recommendation text
    recommendation = {
        "best_scenario": f"Scenario {best_scenario} - {scenarios[best_scenario]['name']}",
        "min_risk_scenario": f"Scenario {min_risk_scenario} - {scenarios[min_risk_scenario]['name']}",
        "max_upside_scenario": f"Scenario {max_upside_scenario} - {scenarios[max_upside_scenario]['name']}",
        "advice": ""
    }
    
    # Add some nuance
    if best_scenario == "A":
        recommendation["advice"] = "Selling BTC provides certainty but may miss out on future BTC appreciation."
    elif best_scenario == "B":
        recommendation["advice"] = f"Borrowing against BTC has a {format_percent(liquidation_prob_b)} chance of liquidation in our simulations."
    elif best_scenario == "C":
        recommendation["advice"] = f"Using BTC as collateral has a {format_percent(liquidation_prob_c)} chance of liquidation, but a {format_percent(debt_payoff_prob_c)} chance of paying off debt completely."
    
    return recommendation

def main():
    st.title("Bitcoin Housing Strategy Analyzer")
    st.markdown("""
    Compare three strategies for using Bitcoin to purchase a house:
    - **Scenario A**: Sell BTC to buy a house
    - **Scenario B**: Borrow against BTC to buy a house
    - **Scenario C**: Use BTC as secondary collateral
    
    This model uses Geometric Brownian Motion (GBM) for stochastic Bitcoin price simulation.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inputs", "Results", "Charts", "Comparison", "Simulations"])
    
    with tab1:
        st.header("Input Parameters")
        
        # Use columns for more compact layout
        col1, col2 = st.columns(2)
        
        # Bitcoin Parameters
        with col1:
            st.subheader("Bitcoin Parameters")
            initial_btc = st.number_input("Initial BTC Holdings", 
                                         min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            
            initial_btc_price = st.number_input("Current BTC Price ($)", 
                                             min_value=1000, max_value=1000000, value=50000, step=1000)
            
            btc_basis_price = st.number_input("BTC Basis Price ($)", 
                                           min_value=1000, max_value=1000000, value=20000, step=1000)
            
            btc_appreciation_rate = st.slider("BTC Expected Annual Return (μ) (%)", 
                                           min_value=-20.0, max_value=100.0, value=20.0, step=1.0) / 100
            
            btc_volatility = st.slider("BTC Volatility (σ)", 
                                     min_value=0.1, max_value=2.0, value=0.6, step=0.1)
            
            btc_selling_fee = st.slider("BTC Selling Fee (%)", 
                                     min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100
        
        # House Parameters
        with col2:
            st.subheader("House Parameters")
            initial_house_price = st.number_input("Initial House Price ($)", 
                                               min_value=50000, max_value=10000000, value=500000, step=10000)
            
            house_appreciation_rate = st.slider("House Appreciation Rate (%)", 
                                             min_value=-5.0, max_value=20.0, value=5.0, step=0.1) / 100
            
            house_purchase_fee = st.slider("House Purchase Fee (%)", 
                                        min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
            
            annual_house_cost = st.slider("Annual House Costs (%)", 
                                       min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
        
        # Financial Parameters
        st.subheader("Financial Parameters")
        col3, col4 = st.columns(2)
        
        with col3:
            capital_gains_tax = st.slider("Capital Gains Tax (%)", 
                                       min_value=0.0, max_value=50.0, value=20.0, step=1.0) / 100
            
            loan_to_value_ratio = st.slider("Loan-to-Value Ratio (%)", 
                                         min_value=10.0, max_value=90.0, value=40.0, step=1.0) / 100
        
        with col4:
            mortgage_rate = st.slider("Mortgage Rate (%)", 
                                    min_value=1.0, max_value=15.0, value=5.0, step=0.1) / 100
            
            inflation_rate = st.slider("Inflation Rate (%)", 
                                     min_value=0.0, max_value=20.0, value=3.0, step=0.1) / 100
        
        # Time and Simulation Parameters
        st.subheader("Time and Simulation Parameters")
        col5, col6 = st.columns(2)
        
        with col5:
            time_horizon = st.slider("Time Horizon (Years)", 
                                   min_value=1, max_value=30, value=10, step=1)
        
        with col6:
            num_simulations = st.slider("Number of Monte Carlo Simulations", 
                                      min_value=100, max_value=2000, value=500, step=100)
        
        # Save/Load inputs
        st.subheader("Save/Load Configuration")
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("Save Current Inputs"):
                inputs = {
                    "initial_btc": initial_btc,
                    "initial_btc_price": initial_btc_price,
                    "btc_basis_price": btc_basis_price,
                    "btc_appreciation_rate": btc_appreciation_rate,
                    "initial_house_price": initial_house_price,
                    "house_appreciation_rate": house_appreciation_rate,
                    "capital_gains_tax": capital_gains_tax,
                    "loan_to_value_ratio": loan_to_value_ratio,
                    "mortgage_rate": mortgage_rate,
                    "time_horizon": time_horizon,
                    "btc_volatility": btc_volatility,
                    "inflation_rate": inflation_rate,
                    "btc_selling_fee": btc_selling_fee,
                    "house_purchase_fee": house_purchase_fee,
                    "annual_house_cost": annual_house_cost,
                    "num_simulations": num_simulations
                }
                
                filename = f"btc_housing_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.markdown(generate_json_download_link(inputs, filename=filename), unsafe_allow_html=True)
        
        with col6:
            uploaded_file = st.file_uploader("Load Inputs", type="json")
            if uploaded_file is not None:
                try:
                    inputs = json.load(uploaded_file)
                    st.success("Inputs loaded successfully! Run analysis to see results.")
                    st.session_state.loaded_inputs = inputs
                except Exception as e:
                    st.error(f"Error loading inputs: {str(e)}")
        
        # Run analysis button
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running Monte Carlo simulations..."):
                # Get input values
                inputs = {
                    "initial_btc": initial_btc,
                    "initial_btc_price": initial_btc_price,
                    "btc_basis_price": btc_basis_price,
                    "btc_appreciation_rate": btc_appreciation_rate,
                    "initial_house_price": initial_house_price,
                    "house_appreciation_rate": house_appreciation_rate,
                    "capital_gains_tax": capital_gains_tax,
                    "loan_to_value_ratio": loan_to_value_ratio,
                    "mortgage_rate": mortgage_rate,
                    "time_horizon": time_horizon,
                    "btc_volatility": btc_volatility,
                    "inflation_rate": inflation_rate,
                    "btc_selling_fee": btc_selling_fee,
                    "house_purchase_fee": house_purchase_fee,
                    "annual_house_cost": annual_house_cost,
                    "num_simulations": num_simulations
                }
                
                # Run analysis
                model, scenario_a, scenario_b, scenario_c, simulations = run_analysis(inputs)
                
                # Store results in session state
                st.session_state.model = model
                st.session_state.scenario_a = scenario_a
                st.session_state.scenario_b = scenario_b
                st.session_state.scenario_c = scenario_c
                st.session_state.simulations = simulations
                
                # Switch to results tab
                st.rerun()

    with tab2:
        if 'scenario_a' in st.session_state:
            st.header("Analysis Results")
            
            # Display metrics for each scenario in columns
            col1, col2, col3 = st.columns(3)
            
            display_scenario_metrics(st.session_state.scenario_a, st.session_state.simulations["Scenario A"], col1)
            display_scenario_metrics(st.session_state.scenario_b, st.session_state.simulations["Scenario B"], col2)
            display_scenario_metrics(st.session_state.scenario_c, st.session_state.simulations["Scenario C"], col3)
            
            # Display recommendation
            st.subheader("Recommendation")
            recommendation = get_recommendation(
                st.session_state.scenario_a,
                st.session_state.scenario_b, 
                st.session_state.scenario_c, 
                st.session_state.simulations
            )
            
            # Use columns for recommendation
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"**Best Option:** {recommendation['best_scenario']}")
                st.write(f"**Lowest Risk:** {recommendation['min_risk_scenario']}")
                st.write(f"**Highest Upside:** {recommendation['max_upside_scenario']}")
                st.write(f"**Consideration:** {recommendation['advice']}")
            
            with col2:
                # Create a small pie chart showing the relative performance
                net_values = [
                    st.session_state.scenario_a["net_value"],
                    st.session_state.scenario_b["net_value"],
                    st.session_state.scenario_c["net_value"]
                ]
                
                # Make sure all values are positive for the pie chart
                if all(value > 0 for value in net_values):
                    fig = go.Figure(data=[go.Pie(
                        labels=['Scenario A', 'Scenario B', 'Scenario C'],
                        values=net_values,
                        marker_colors=['#3498db', '#2ecc71', '#e74c3c'],
                        textinfo='percent'
                    )])
                    
                    fig.update_layout(height=250, width=250, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("Run the analysis in the Inputs tab to see results")
    
    with tab3:
        if 'simulations' in st.session_state:
            st.header("Visualization Charts")
            
            # Create comparison charts
            comparison_chart = create_comparison_chart(
                st.session_state.scenario_a,
                st.session_state.scenario_b,
                st.session_state.scenario_c,
                st.session_state.simulations
            )
            
            bull_bear_chart = create_bull_bear_chart(st.session_state.simulations)
            distribution_chart = create_distribution_chart(st.session_state.simulations)
            
            # Display comparison charts
            st.subheader("Scenario Comparison")
            st.plotly_chart(comparison_chart, use_container_width=True)
            st.plotly_chart(bull_bear_chart, use_container_width=True)
            st.plotly_chart(distribution_chart, use_container_width=True)
            
            # Create scenario-specific charts
            scenario_charts = create_scenario_charts(
                st.session_state.scenario_a,
                st.session_state.scenario_b,
                st.session_state.scenario_c
            )
            
            # Display scenario-specific charts
            st.subheader("Scenario-Specific Charts")
            
            if "ltv_chart" in scenario_charts:
                st.plotly_chart(scenario_charts["ltv_chart"], use_container_width=True)
            
            if "debt_chart" in scenario_charts:
                st.plotly_chart(scenario_charts["debt_chart"], use_container_width=True)
            
            if "collateral_chart" in scenario_charts:
                st.plotly_chart(scenario_charts["collateral_chart"], use_container_width=True)
        else:
            st.info("Run the analysis in the Inputs tab to see charts")
    
    with tab4:
        if 'scenario_a' in st.session_state:
            st.header("Scenario Comparison")
            
            # Create a dataframe for comparison
            comparison_data = {
                "Metric": [
                    "Net Value",
                    "Total Return",
                    "Final House Value",
                    "Final BTC Value",
                    "Bear Case (10%)",
                    "Base Case (50%)",
                    "Bull Case (90%)",
                    "Liquidation Risk"
                ],
                "Scenario A": [
                    format_currency(st.session_state.scenario_a["net_value"]),
                    format_percent(st.session_state.scenario_a["total_return"]),
                    format_currency(st.session_state.scenario_a["final_house_value"]),
                    format_currency(st.session_state.scenario_a["final_btc_value_if_held"]) + " (if held)",
                    format_currency(st.session_state.simulations["Scenario A"]["Bear Case"]),
                    format_currency(st.session_state.simulations["Scenario A"]["Base Case"]),
                    format_currency(st.session_state.simulations["Scenario A"]["Bull Case"]),
                    "None"
                ],
                "Scenario B": [
                    format_currency(st.session_state.scenario_b["net_value"]),
                    format_percent(st.session_state.scenario_b["total_return"]),
                    format_currency(st.session_state.scenario_b["final_house_value"]),
                    format_currency(st.session_state.scenario_b["final_btc_value"]),
                    format_currency(st.session_state.simulations["Scenario B"]["Bear Case"]),
                    format_currency(st.session_state.simulations["Scenario B"]["Base Case"]),
                    format_currency(st.session_state.simulations["Scenario B"]["Bull Case"]),
                    format_percent(st.session_state.scenario_b.get("liquidation_probability", 0))
                ],
                "Scenario C": [
                    format_currency(st.session_state.scenario_c["net_value"]),
                    format_percent(st.session_state.scenario_c["total_return"]),
                    format_currency(st.session_state.scenario_c["final_house_value"]),
                    format_currency(st.session_state.scenario_c["final_btc_value"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Bear Case"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Base Case"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Bull Case"]),
                    format_percent(st.session_state.scenario_c.get("liquidation_probability", 0))
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Show a radar chart for visual comparison
            st.subheader("Strategy Comparison Radar Chart")
            
            # Normalize values for radar chart
            radar_data = {
                "category": [
                    "Net Value",
                    "Total Return",
                    "House Value",
                    "BTC Value",
                    "Bear Case",
                    "Base Case",
                    "Bull Case",
                    "Risk Avoidance"
                ],
                "Scenario A": [
                    st.session_state.scenario_a["net_value"],
                    st.session_state.scenario_a["total_return"],
                    st.session_state.scenario_a["final_house_value"],
                    st.session_state.scenario_a["final_btc_value_if_held"],
                    st.session_state.simulations["Scenario A"]["Bear Case"],
                    st.session_state.simulations["Scenario A"]["Base Case"],
                    st.session_state.simulations["Scenario A"]["Bull Case"],
                    100  # No liquidation risk
                ],
                "Scenario B": [
                    st.session_state.scenario_b["net_value"],
                    st.session_state.scenario_b["total_return"],
                    st.session_state.scenario_b["final_house_value"],
                    st.session_state.scenario_b["final_btc_value"],
                    st.session_state.simulations["Scenario B"]["Bear Case"],
                    st.session_state.simulations["Scenario B"]["Base Case"],
                    st.session_state.simulations["Scenario B"]["Bull Case"],
                    100 - st.session_state.scenario_b.get("liquidation_probability", 0)  # Invert liquidation probability
                ],
                "Scenario C": [
                    st.session_state.scenario_c["net_value"],
                    st.session_state.scenario_c["total_return"],
                    st.session_state.scenario_c["final_house_value"],
                    st.session_state.scenario_c["final_btc_value"],
                    st.session_state.simulations["Scenario C"]["Bear Case"],
                    st.session_state.simulations["Scenario C"]["Base Case"],
                    st.session_state.simulations["Scenario C"]["Bull Case"],
                    100 - st.session_state.scenario_c.get("liquidation_probability", 0)  # Invert liquidation probability
                ]
            }
            
            # Normalize each category to 0-100 scale
            for category in radar_data["category"]:
                if category == "Risk Avoidance":
                    continue  # Already normalized
                
                max_val = max(radar_data["Scenario A"][radar_data["category"].index(category)],
                             radar_data["Scenario B"][radar_data["category"].index(category)],
                             radar_data["Scenario C"][radar_data["category"].index(category)])
                
                if max_val > 0:
                    idx = radar_data["category"].index(category)
                    radar_data["Scenario A"][idx] = (radar_data["Scenario A"][idx] / max_val) * 100
                    radar_data["Scenario B"][idx] = (radar_data["Scenario B"][idx] / max_val) * 100
                    radar_data["Scenario C"][idx] = (radar_data["Scenario C"][idx] / max_val) * 100
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=radar_data["Scenario A"],
                theta=radar_data["category"],
                fill='toself',
                name='Scenario A',
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=radar_data["Scenario B"],
                theta=radar_data["category"],
                fill='toself',
                name='Scenario B',
                marker_color='#2ecc71'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=radar_data["Scenario C"],
                theta=radar_data["category"],
                fill='toself',
                name='Scenario C',
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the analysis in the Inputs tab to see comparison")
    
    with tab5:
        if 'model' in st.session_state:
            st.header("Monte Carlo Simulations")
            
            # Display BTC price path chart
            st.subheader("Bitcoin Price Paths (Geometric Brownian Motion)")
            btc_price_chart = create_btc_price_path_chart(st.session_state.model)
            st.plotly_chart(btc_price_chart, use_container_width=True)
            
            # Explanation of GBM
            with st.expander("About Geometric Brownian Motion (GBM)"):
                st.markdown("""
                **Geometric Brownian Motion (GBM)** is a continuous-time stochastic process often used to model stock prices, cryptocurrencies, and other financial assets. The model is defined by the stochastic differential equation:
                
                dS = μS dt + σS dW
                
                Where:
                - S is the asset price
                - μ is the drift (expected return)
                - σ is the volatility
                - dW is a Wiener process (Brownian motion)
                
                GBM assumes that:
                1. Price changes are independent of the past (Markov property)
                2. Returns are normally distributed
                3. Volatility is constant
                
                While these assumptions may not perfectly hold for Bitcoin, GBM provides a reasonable approximation for modeling future price scenarios, especially for risk assessment.
                """)
            
            # Display probability metrics
            st.subheader("Probability Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scenario B liquidation probability
                liquidation_prob_b = st.session_state.scenario_b.get("liquidation_probability", 0)
                st.metric("Scenario B Liquidation Probability", format_percent(liquidation_prob_b))
                
                # BTC price doubled probability
                btc_price_paths = st.session_state.model.btc_price_paths
                final_btc_prices = btc_price_paths[:, -1]
                doubled_prob = np.mean(final_btc_prices > 2 * st.session_state.model.P0) * 100
                st.metric("Probability BTC Price Doubles", format_percent(doubled_prob))
            
            with col2:
                # Scenario C metrics
                liquidation_prob_c = st.session_state.scenario_c.get("liquidation_probability", 0)
                st.metric("Scenario C Liquidation Probability", format_percent(liquidation_prob_c))
                
                debt_payoff_prob = st.session_state.scenario_c.get("debt_payoff_probability", 0)
                st.metric("Scenario C Debt Payoff Probability", format_percent(debt_payoff_prob))
            
            # Display distribution of final BTC prices
            st.subheader("Distribution of Final BTC Prices")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=final_btc_prices,
                nbinsx=30,
                marker_color='#f39c12'
            ))
            
            # Add vertical line for initial price
            initial_price = st.session_state.model.P0
            fig.add_shape(
                type="line",
                x0=initial_price, y0=0,
                x1=initial_price, y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=initial_price,
                y=0.95,
                yref="paper",
                text="Initial Price",
                showarrow=True,
                arrowhead=1
            )
            
            fig.update_layout(
                title='Distribution of Final BTC Prices',
                xaxis_title='BTC Price ($)',
                xaxis=dict(tickformat='$,.0f'),
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the analysis in the Inputs tab to see simulation details")
    
    # Add footer
    st.markdown("---")
    st.markdown("Bitcoin Housing Strategy Analyzer - Using Geometric Brownian Motion to model Bitcoin price uncertainty")

if __name__ == "__main__":
    main()
