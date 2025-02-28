import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import base64
import io
import os

# -------------------------
# Configure Streamlit Page & Dark Mode
# -------------------------
st.set_page_config(
    page_title="Bitcoin Housing Strategy Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for Dark Mode
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stButton button {
        background-color: #2ecc71;
        color: #121212;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Model Definition
# -------------------------
class BTCHousingModel:
    """
    Model for comparing three scenarios of using Bitcoin for a house purchase:
      A) Sell BTC to buy house
      B) Borrow against BTC (using profit only) to buy house
      C) Use BTC (profit only) as secondary collateral for a self-paying mortgage
    
    The model uses historical data (if provided) to calibrate Geometric Brownian Motion (GBM) parameters.
    """
    
    def __init__(self, 
                 initial_btc,              # B₀: Initial BTC holdings
                 initial_btc_price,        # P₀: Current BTC price
                 btc_basis_price,          # Price at which BTC was acquired (basis)
                 btc_appreciation_rate,    # μ: Expected annual BTC return (if no historical data)
                 initial_house_price,      # H₀: House price
                 house_appreciation_rate,  # rₕ: House appreciation rate
                 capital_gains_tax,        # τ: Capital gains tax rate
                 mortgage_rate,            # iₛₜ: Mortgage interest rate
                 time_horizon,             # T: Time horizon (years)
                 btc_volatility,           # σ: BTC volatility (if no historical data)
                 inflation_rate,           # π: Inflation rate
                 btc_selling_fee,          # Fee for selling BTC (as fraction)
                 house_purchase_fee,       # House purchase fee (as fraction)
                 annual_house_cost,        # Annual house cost (as fraction)
                 num_simulations,          # Number of Monte Carlo simulations
                 time_steps_per_year=12,   # Monthly steps by default
                 historical_btc_data=None  # Historical BTC data (pandas DataFrame expected)
                ):
        
        self.B0 = initial_btc
        self.P0 = initial_btc_price
        self.P_basis = btc_basis_price
        self.mu = btc_appreciation_rate    # Will be overridden if historical data is provided
        self.H0 = initial_house_price
        self.rh = house_appreciation_rate
        self.tau = capital_gains_tax
        # LTV is now hardcoded to 125%
        self.LTV = 1.25  
        self.i_st = mortgage_rate
        self.T = time_horizon
        self.sigma = btc_volatility        # Default volatility; may be updated via historical data
        self.pi = inflation_rate
        self.F_BTC = btc_selling_fee
        self.F_House = house_purchase_fee
        self.f_House = annual_house_cost
        self.num_simulations = num_simulations
        self.time_steps_per_year = time_steps_per_year
        self.historical_btc_data = historical_btc_data
        
        # Derived values
        self.initial_btc_value = self.B0 * self.P0
        self.total_steps = int(self.T * self.time_steps_per_year)
        self.dt = 1.0 / self.time_steps_per_year
        
        # Pre-generate price paths
        self.btc_price_paths = self._generate_btc_price_paths()
        self.house_price_paths = self._generate_house_price_paths()
    
    def _load_historical_btc_data(self):
        """
        If historical_btc_data is provided (as a DataFrame), extract the 'Close' price column.
        Otherwise, return None.
        """
        if self.historical_btc_data is not None:
            if 'Close' in self.historical_btc_data.columns:
                # Sort by date if available
                if 'Timestamp' in self.historical_btc_data.columns:
                    self.historical_btc_data.sort_values('Timestamp', inplace=True)
                return self.historical_btc_data['Close'].values
        return None

    def _generate_btc_price_paths(self):
        """
        Generate BTC price paths using Geometric Brownian Motion (GBM).
        If historical data is available, calibrate drift and volatility.
        Returns a 2D numpy array of shape (num_simulations, total_steps+1).
        """
        # Initialize price paths
        price_paths = np.zeros((self.num_simulations, self.total_steps + 1))
        price_paths[:, 0] = self.P0
        
        # Check for historical data and calibrate parameters
        historical_prices = self._load_historical_btc_data()
        if historical_prices is not None and len(historical_prices) > 1:
            # Calculate log returns
            historical_returns = np.diff(np.log(historical_prices))
            # Annualized drift and volatility (assuming daily data)
            hist_drift = np.mean(historical_returns) * 365
            hist_volatility = np.std(historical_returns) * np.sqrt(365)
            # Update parameters if reasonable
            if 0.1 <= hist_volatility <= 2.0:
                self.mu = hist_drift
                self.sigma = hist_volatility
                st.write(f"Using historical calibration: drift = {self.mu:.2f}, volatility = {self.sigma:.2f}")
        
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        # Generate random shocks using antithetic variates for variance reduction
        half_sims = self.num_simulations // 2
        Z = np.random.normal(0, 1, (half_sims, self.total_steps))
        Z_antithetic = -Z
        Z = np.vstack((Z, Z_antithetic))
        if Z.shape[0] < self.num_simulations:
            extra = np.random.normal(0, 1, (self.num_simulations - Z.shape[0], self.total_steps))
            Z = np.vstack((Z, extra))
        
        for t in range(1, self.total_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
        return price_paths

    def _generate_house_price_paths(self):
        """
        Generate house price paths using GBM with lower volatility.
        """
        house_volatility = self.rh * 0.3  # House prices are less volatile
        price_paths = np.zeros((self.num_simulations, self.total_steps + 1))
        price_paths[:, 0] = self.H0
        
        Z = np.random.normal(0, 1, (self.num_simulations, self.total_steps))
        drift = (self.rh - 0.5 * house_volatility**2) * self.dt
        diffusion = house_volatility * np.sqrt(self.dt)
        
        for t in range(1, self.total_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
        return price_paths

    def _get_yearly_indices(self):
        """Return indices corresponding to yearly steps."""
        return [i * self.time_steps_per_year for i in range(self.T + 1)]

    def scenario_a_sell_btc(self):
        """
        Scenario A: Sell BTC (using profit only) to buy a house.
        Profit is defined as: profit = B₀*(P₀ - P_basis)
        """
        net_values = np.zeros(self.num_simulations)
        btc_values_if_held = np.zeros(self.num_simulations)
        house_cost = self.H0 * (1 + self.F_House)
        
        for sim in range(self.num_simulations):
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            # Compute profit (only gains above basis count)
            profit = self.B0 * max(0, self.P0 - self.P_basis)
            btc_selling_fee = profit * self.F_BTC
            capital_gains_tax = profit * self.tau
            net_proceeds = profit - btc_selling_fee - capital_gains_tax
            
            can_afford = net_proceeds >= house_cost
            remaining_cash = net_proceeds - house_cost if can_afford else 0
            
            if not can_afford:
                net_values[sim] = self.B0 * btc_prices[-1]  # hold BTC if house purchase not feasible
                btc_values_if_held[sim] = self.B0 * btc_prices[-1]
                continue
            
            # Compute present value of annual house costs
            yearly_indices = self._get_yearly_indices()
            holding_costs_pv = 0
            for i, idx in enumerate(yearly_indices):
                if i == 0:
                    continue
                yearly_cost = self.f_House * house_prices[idx]
                holding_costs_pv += yearly_cost / ((1 + self.pi) ** i)
            
            final_house_value = house_prices[-1]
            final_btc_value_if_held = self.B0 * btc_prices[-1]
            net_value = final_house_value - house_cost - holding_costs_pv + remaining_cash
            
            net_values[sim] = net_value
            btc_values_if_held[sim] = final_btc_value_if_held
        
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        btc_held_percentiles = np.percentile(btc_values_if_held, percentiles)
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        opportunity_cost = btc_values_if_held[median_sim_idx] - (self.B0 * self.P0)
        
        return {
            "scenario": "A - Sell BTC to Buy House",
            "can_afford_house": net_proceeds >= house_cost,
            "initial_btc_value": self.B0 * self.P0,
            "net_proceeds_after_tax": net_proceeds,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "holding_costs_pv": holding_costs_pv,
            "final_house_value": self.house_price_paths[median_sim_idx, -1],
            "final_btc_value_if_held": btc_values_if_held[median_sim_idx],
            "net_value": net_values[median_sim_idx],
            "opportunity_cost": opportunity_cost,
            "total_return": (net_values[median_sim_idx] / house_cost) * 100 if house_cost > 0 else 0,
            "btc_return_if_held": ((btc_values_if_held[median_sim_idx] / (self.B0 * self.P0)) - 1) * 100,
            "net_value_percentiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], net_value_percentiles)),
            "btc_held_percentiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], btc_held_percentiles)),
            "all_net_values": net_values.tolist(),
            "all_btc_values": btc_values_if_held.tolist()
        }
    
    def scenario_b_borrow_against_btc(self):
        """
        Scenario B: Borrow against BTC profit to buy a house.
        Loan amount is calculated as: loan_amount = LTV * profit, where LTV is fixed at 125%.
        """
        net_values = np.zeros(self.num_simulations)
        liquidation_occurred = np.zeros(self.num_simulations, dtype=bool)
        liquidation_times = np.full(self.num_simulations, np.nan)
        
        # Use profit only for borrowing
        profit = self.B0 * max(0, self.P0 - self.P_basis)
        loan_amount = self.LTV * profit
        house_cost = self.H0 * (1 + self.F_House)
        can_afford = loan_amount >= house_cost
        remaining_cash = loan_amount - house_cost if can_afford else 0
        
        if not can_afford:
            return {
                "scenario": "B - Borrow Against BTC",
                "can_afford_house": False,
                "loan_amount": loan_amount,
                "house_cost": house_cost,
                "net_value": 0,
                "total_return": 0
            }
        
        # Set liquidation threshold (if loan-to-profit ratio exceeds 125% of profit value)
        liquidation_threshold = 1.25  
        for sim in range(self.num_simulations):
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            loan_values = np.zeros(self.total_steps + 1)
            loan_values[0] = loan_amount
            
            for t in range(1, self.total_steps + 1):
                loan_values[t] = loan_values[t-1] * (1 + self.i_st/self.time_steps_per_year)
            
            # For BTC profit, update profit over time:
            btc_profit_values = self.B0 * np.maximum(0, btc_prices - self.P_basis)
            current_ltv = loan_values / np.maximum(btc_profit_values, 1e-6)
            liquidation_points = np.where(current_ltv > liquidation_threshold)[0]
            
            if len(liquidation_points) > 0:
                liquidation_occurred[sim] = True
                liquidation_times[sim] = liquidation_points[0] / self.time_steps_per_year
                final_btc_value = 0
            else:
                final_btc_value = btc_profit_values[-1]
            
            final_house_value = house_prices[-1]
            final_loan_value = loan_values[-1]
            net_values[sim] = final_house_value + final_btc_value - final_loan_value + remaining_cash
        
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        median_btc_prices = self.btc_price_paths[median_sim_idx]
        median_btc_profit = self.B0 * np.maximum(0, median_btc_prices - self.P_basis)
        
        median_loan_values = np.zeros(self.total_steps + 1)
        median_loan_values[0] = loan_amount
        for t in range(1, self.total_steps + 1):
            median_loan_values[t] = median_loan_values[t-1] * (1 + self.i_st/self.time_steps_per_year)
        
        # For reporting, convert LTV history to percentage
        median_ltv = (median_loan_values / np.maximum(median_btc_profit, 1e-6)) * 100
        time_points = np.linspace(0, self.T, self.total_steps + 1)
        liquidation_probability = np.mean(liquidation_occurred) * 100
        
        return {
            "scenario": "B - Borrow Against BTC",
            "can_afford_house": can_afford,
            "loan_amount": loan_amount,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "loan_with_interest": median_loan_values[-1],
            "final_btc_price": median_btc_prices[-1],
            "final_btc_value": median_btc_profit[-1] if not liquidation_occurred[median_sim_idx] else 0,
            "final_house_value": self.house_price_paths[median_sim_idx, -1],
            "liquidation_occurred": liquidation_occurred[median_sim_idx],
            "liquidation_time": liquidation_times[median_sim_idx] if liquidation_occurred[median_sim_idx] else None,
            "net_value": net_values[median_sim_idx],
            "total_return": (net_values[median_sim_idx] / (self.B0 * self.P0)) * 100,
            "time_points": time_points.tolist(),
            "btc_value_history": median_btc_profit.tolist(),
            "loan_value_history": median_loan_values.tolist(),
            "ltv_history": median_ltv.tolist(),
            "liquidation_probability": liquidation_probability,
            "net_value_percentiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], net_value_percentiles)),
            "all_net_values": net_values.tolist()
        }
    
    def scenario_c_btc_collateral(self):
        """
        Scenario C: Use BTC (profit only) as secondary collateral.
        BTC remains as an asset used to accelerate mortgage payments.
        """
        net_values = np.zeros(self.num_simulations)
        btc_liquidated = np.zeros(self.num_simulations, dtype=bool)
        liquidation_times = np.full(self.num_simulations, np.nan)
        debt_paid_off = np.zeros(self.num_simulations, dtype=bool)
        
        initial_mortgage = self.H0 * (1 + self.F_House)
        yearly_indices = self._get_yearly_indices()
        
        # For storing the median simulation history
        median_debt_history = None
        median_house_value_history = None
        median_btc_profit_history = None
        median_payments_from_btc = None
        
        for sim in range(self.num_simulations):
            btc_prices = self.btc_price_paths[sim]
            house_prices = self.house_price_paths[sim]
            
            debt_history = np.zeros(len(yearly_indices))
            house_value_history = np.zeros(len(yearly_indices))
            btc_profit_history = np.zeros(len(yearly_indices))
            payments_from_btc = np.zeros(len(yearly_indices))
            
            debt_remaining = initial_mortgage
            debt_history[0] = debt_remaining
            house_value_history[0] = house_prices[0]
            # Use profit only: if current price is below basis, profit is zero
            btc_profit_history[0] = self.B0 * max(0, btc_prices[0] - self.P_basis)
            
            sim_btc_liquidated = False
            sim_liquidation_time = None
            
            for t in range(1, len(yearly_indices)):
                idx = yearly_indices[t]
                house_value = house_prices[idx]
                house_value_history[t] = house_value
                
                if not sim_btc_liquidated:
                    current_profit = self.B0 * max(0, btc_prices[idx] - self.P_basis)
                else:
                    current_profit = 0
                btc_profit_history[t] = current_profit
                
                years_remaining = self.T - t + 1
                if years_remaining > 0 and debt_remaining > 0:
                    regular_payment = debt_remaining * self.i_st * (1 + self.i_st)**years_remaining / ((1 + self.i_st)**years_remaining - 1)
                else:
                    regular_payment = debt_remaining
                
                debt_with_interest = debt_remaining * (1 + self.i_st)
                
                if t > 1 and not sim_btc_liquidated:
                    btc_appreciation = current_profit - btc_profit_history[t-1]
                    usable_btc_gain = max(0, btc_appreciation * 0.5)
                else:
                    usable_btc_gain = 0
                
                additional_payment = min(usable_btc_gain, debt_with_interest - regular_payment)
                payments_from_btc[t] = additional_payment
                total_payment = regular_payment + additional_payment
                debt_remaining = max(0, debt_with_interest - total_payment)
                debt_history[t] = debt_remaining
                
                if debt_remaining <= 0:
                    debt_remaining = 0
                    debt_history[t:] = 0
                    debt_paid_off[sim] = True
                    break
                
                if not sim_btc_liquidated and t > 1:
                    peak_profit = np.max(btc_profit_history[:t])
                    if current_profit < 0.3 * peak_profit:
                        sim_btc_liquidated = True
                        sim_liquidation_time = t
                        current_profit = 0
                        btc_profit_history[t:] = 0
            
            btc_liquidated[sim] = sim_btc_liquidated
            if sim_liquidation_time is not None:
                liquidation_times[sim] = sim_liquidation_time
            
            final_house_value = house_prices[-1]
            final_btc_profit = self.B0 * max(0, btc_prices[-1] - self.P_basis) if not sim_btc_liquidated else 0
            net_value = final_house_value + final_btc_profit - debt_remaining
            net_values[sim] = net_value
            
            if median_debt_history is None:
                median_debt_history = debt_history
                median_house_value_history = house_value_history
                median_btc_profit_history = btc_profit_history
                median_payments_from_btc = payments_from_btc
        
        median_sim_idx = np.argsort(net_values)[len(net_values)//2]
        time_points = np.arange(self.T + 1)
        liquidation_probability = np.mean(btc_liquidated) * 100
        debt_payoff_probability = np.mean(debt_paid_off) * 100
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        
        return {
            "scenario": "C - BTC as Secondary Collateral",
            "initial_loan": initial_mortgage,
            "final_house_value": self.house_price_paths[median_sim_idx, -1],
            "final_btc_value": self.B0 * max(0, self.btc_price_paths[median_sim_idx, -1] - self.P_basis) if not btc_liquidated[median_sim_idx] else 0,
            "final_debt": median_debt_history[-1] if median_debt_history is not None else 0,
            "liquidation_occurred": btc_liquidated[median_sim_idx],
            "liquidation_time": liquidation_times[median_sim_idx] if btc_liquidated[median_sim_idx] else None,
            "debt_paid_off": debt_paid_off[median_sim_idx],
            "debt_history": median_debt_history.tolist() if median_debt_history is not None else [],
            "house_value_history": median_house_value_history.tolist() if median_house_value_history is not None else [],
            "btc_value_history": median_btc_profit_history.tolist() if median_btc_profit_history is not None else [],
            "payments_from_btc": median_payments_from_btc.tolist() if median_payments_from_btc is not None else [],
            "time_points": time_points.tolist(),
            "net_value": net_values[median_sim_idx],
            "total_return": (net_values[median_sim_idx] / (self.B0 * self.P0)) * 100,
            "liquidation_probability": liquidation_probability,
            "debt_payoff_probability": debt_payoff_probability,
            "net_value_percentiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], net_value_percentiles)),
            "all_net_values": net_values.tolist()
        }
    
    def simulate_scenarios(self):
        """
        Run all three scenarios and return aggregated results.
        """
        scenario_a = self.scenario_a_sell_btc()
        scenario_b = self.scenario_b_borrow_against_btc()
        scenario_c = self.scenario_c_btc_collateral()
        
        scenario_a_results = np.array(scenario_a["all_net_values"])
        scenario_b_results = np.array(scenario_b["all_net_values"]) if "all_net_values" in scenario_b else np.zeros(self.num_simulations)
        scenario_c_results = np.array(scenario_c["all_net_values"])
        
        percentiles = [10, 50, 90]
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

# -------------------------
# Utility Functions for Formatting and Downloads
# -------------------------
def format_currency(value):
    return f"${value:,.2f}"

def format_percent(value):
    return f"{value:.2f}%"

def generate_json_download_link(json_data, filename="btc_housing_inputs.json"):
    json_str = json.dumps(json_data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:text/json;base64,{b64}" download="{filename}">Download JSON</a>'
    return href

# -------------------------
# Main Analysis Runner
# -------------------------
def run_analysis(inputs):
    historical_btc_data = inputs.get("historical_btc_data", None)
    model = BTCHousingModel(
        initial_btc=inputs["initial_btc"],
        initial_btc_price=inputs["initial_btc_price"],
        btc_basis_price=inputs["btc_basis_price"],
        btc_appreciation_rate=inputs["btc_appreciation_rate"],
        initial_house_price=inputs["initial_house_price"],
        house_appreciation_rate=inputs["house_appreciation_rate"],
        capital_gains_tax=inputs["capital_gains_tax"],
        mortgage_rate=inputs["mortgage_rate"],
        time_horizon=inputs["time_horizon"],
        btc_volatility=inputs["btc_volatility"],
        inflation_rate=inputs["inflation_rate"],
        btc_selling_fee=inputs["btc_selling_fee"],
        house_purchase_fee=inputs["house_purchase_fee"],
        annual_house_cost=inputs["annual_house_cost"],
        num_simulations=inputs["num_simulations"],
        historical_btc_data=historical_btc_data
    )
    
    scenario_a = model.scenario_a_sell_btc()
    scenario_b = model.scenario_b_borrow_against_btc()
    scenario_c = model.scenario_c_btc_collateral()
    simulations = model.simulate_scenarios()
    
    return model, scenario_a, scenario_b, scenario_c, simulations

# -------------------------
# Streamlit App Layout
# -------------------------
def main():
    st.title("Bitcoin Housing Strategy Analyzer")
    st.markdown("""
    **Overview:**  
    This tool analyzes three strategies for using Bitcoin profit (the difference between current price and your BTC basis) in a house purchase.  
    **Strategies:**  
    - **Scenario A:** Sell BTC (profit only) to buy a house  
    - **Scenario B:** Borrow against BTC profit (LTV fixed at 125%) to buy a house  
    - **Scenario C:** Use BTC profit as secondary collateral (self-paying mortgage)  
    """)
    
    st.markdown("**Note:** Loan-to-Value (LTV) ratio is fixed at 125% and cannot be changed.")
    
    # -------------------------
    # Historical Data Input Section
    # -------------------------
    st.subheader("Step 1: Provide Historical BTC Data (Optional)")
    uploaded_file = st.file_uploader("Upload Historical BTC Data CSV (must include a 'Close' column)", type="csv")
    historical_data = None
    if uploaded_file is not None:
        try:
            historical_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded historical data with {len(historical_data)} records.")
            st.dataframe(historical_data.head(10))
        except Exception as e:
            st.error(f"Error loading historical data: {e}")
    
    # -------------------------
    # Input Parameters (All via input boxes)
    # -------------------------
    st.subheader("Step 2: Enter Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_btc = st.number_input("Initial BTC Holdings", value=1.0, step=0.1)
        initial_btc_price = st.number_input("Current BTC Price ($)", value=50000.0, step=1000.0)
        btc_basis_price = st.number_input("BTC Basis Price ($)", value=20000.0, step=1000.0)
        btc_appreciation_rate = st.number_input("BTC Expected Annual Return (%)", value=20.0, step=1.0) / 100
        btc_volatility = st.number_input("BTC Volatility (σ, as a decimal)", value=0.6, step=0.1)
        btc_selling_fee = st.number_input("BTC Selling Fee (%)", value=1.0, step=0.1) / 100
        
    with col2:
        initial_house_price = st.number_input("Initial House Price ($)", value=500000.0, step=10000.0)
        house_appreciation_rate = st.number_input("House Appreciation Rate (%)", value=5.0, step=0.1) / 100
        house_purchase_fee = st.number_input("House Purchase Fee (%)", value=2.0, step=0.1) / 100
        annual_house_cost = st.number_input("Annual House Cost (%)", value=2.0, step=0.1) / 100
        capital_gains_tax = st.number_input("Capital Gains Tax (%)", value=20.0, step=1.0) / 100
        mortgage_rate = st.number_input("Mortgage Rate (%)", value=5.0, step=0.1) / 100
    
    st.subheader("Additional Parameters")
    time_horizon = int(st.number_input("Time Horizon (Years)", value=10, step=1))
    num_simulations = int(st.number_input("Number of Monte Carlo Simulations", value=500, step=100))
    inflation_rate = st.number_input("Inflation Rate (%)", value=3.0, step=0.1) / 100
    
    # Save/Load Configuration (optional)
    st.subheader("Save/Load Configuration")
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save Current Inputs"):
            inputs = {
                "initial_btc": initial_btc,
                "initial_btc_price": initial_btc_price,
                "btc_basis_price": btc_basis_price,
                "btc_appreciation_rate": btc_appreciation_rate,
                "btc_volatility": btc_volatility,
                "btc_selling_fee": btc_selling_fee,
                "initial_house_price": initial_house_price,
                "house_appreciation_rate": house_appreciation_rate,
                "house_purchase_fee": house_purchase_fee,
                "annual_house_cost": annual_house_cost,
                "capital_gains_tax": capital_gains_tax,
                "mortgage_rate": mortgage_rate,
                "time_horizon": time_horizon,
                "num_simulations": num_simulations,
                "inflation_rate": inflation_rate,
                "historical_btc_data": historical_data
            }
            filename = f"btc_housing_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.markdown(generate_json_download_link(inputs, filename=filename), unsafe_allow_html=True)
    
    with col_load:
        uploaded_config = st.file_uploader("Load Inputs JSON", type="json", key="config")
        if uploaded_config is not None:
            try:
                inputs = json.load(uploaded_config)
                st.success("Inputs loaded successfully!")
                st.session_state.loaded_inputs = inputs
            except Exception as e:
                st.error(f"Error loading inputs: {e}")
    
    # -------------------------
    # Run Analysis Button
    # -------------------------
    if st.button("Run Analysis"):
        with st.spinner("Running simulations..."):
            inputs = {
                "initial_btc": initial_btc,
                "initial_btc_price": initial_btc_price,
                "btc_basis_price": btc_basis_price,
                "btc_appreciation_rate": btc_appreciation_rate,
                "btc_volatility": btc_volatility,
                "btc_selling_fee": btc_selling_fee,
                "initial_house_price": initial_house_price,
                "house_appreciation_rate": house_appreciation_rate,
                "house_purchase_fee": house_purchase_fee,
                "annual_house_cost": annual_house_cost,
                "capital_gains_tax": capital_gains_tax,
                "mortgage_rate": mortgage_rate,
                "time_horizon": time_horizon,
                "num_simulations": num_simulations,
                "inflation_rate": inflation_rate,
                "historical_btc_data": historical_data
            }
            model, scenario_a, scenario_b, scenario_c, simulations = run_analysis(inputs)
            st.session_state.model = model
            st.session_state.scenario_a = scenario_a
            st.session_state.scenario_b = scenario_b
            st.session_state.scenario_c = scenario_c
            st.session_state.simulations = simulations
            st.success("Analysis complete!")
    
    # -------------------------
    # Display Results (if available)
    # -------------------------
    if 'scenario_a' in st.session_state:
        st.subheader("Analysis Results")
        st.write("### Scenario A - Sell BTC to Buy House")
        st.write(f"Net Value: {format_currency(st.session_state.scenario_a['net_value'])}")
        st.write(f"Total Return: {format_percent(st.session_state.scenario_a['total_return'])}")
        
        st.write("### Scenario B - Borrow Against BTC")
        st.write(f"Net Value: {format_currency(st.session_state.scenario_b['net_value'])}")
        st.write(f"Liquidation Probability: {format_percent(st.session_state.scenario_b.get('liquidation_probability', 0))}")
        
        st.write("### Scenario C - BTC as Secondary Collateral")
        st.write(f"Net Value: {format_currency(st.session_state.scenario_c['net_value'])}")
        st.write(f"Liquidation Probability: {format_percent(st.session_state.scenario_c.get('liquidation_probability', 0))}")
    
    st.markdown("---")
    st.markdown("Bitcoin Housing Strategy Analyzer © " + datetime.now().strftime("%Y"))

if __name__ == "__main__":
    main()
