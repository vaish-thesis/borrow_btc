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
    """
    
    def __init__(self, 
                 initial_btc,              # B₀: Initial BTC holdings
                 current_btc_price,        # P₀: Current market price of BTC at time of analysis
                 btc_purchase_price,       # Price at which BTC was originally purchased (for tax calculations)
                 btc_appreciation_rate,    # rₐ: BTC annual appreciation rate
                 initial_house_price,      # H₀: Initial house price (USD)
                 house_appreciation_rate,  # rₕ: Rate of house appreciation
                 capital_gains_tax,        # τ: Capital gains tax rate
                 loan_to_value_ratio,      # LTV: Loan to value ratio
                 mortgage_rate,            # iₛₜ: Mortgage rate
                 time_horizon,             # T: Time horizon (years)
                 btc_volatility,           # σₐ: BTC volatility
                 inflation_rate,           # π: Inflation rate
                 btc_selling_fee,          # F_BTC: Fee for selling BTC (%)
                 house_purchase_fee,       # F_House: House purchase fee (%)
                 annual_house_cost         # f_House: Annual recurring house cost (%)
                ):
        
        # Store parameters
        self.B0 = initial_btc
        self.P0 = current_btc_price
        self.P_basis = btc_purchase_price
        self.rb = btc_appreciation_rate
        self.H0 = initial_house_price
        self.rh = house_appreciation_rate
        self.tau = capital_gains_tax
        self.LTV = loan_to_value_ratio
        self.i_st = mortgage_rate
        self.T = time_horizon
        self.sigma_b = btc_volatility
        self.pi = inflation_rate
        self.F_BTC = btc_selling_fee
        self.F_House = house_purchase_fee
        self.f_House = annual_house_cost
        
        # Derived values
        self.initial_btc_value = self.B0 * self.P0
        
    def scenario_a_sell_btc(self):
        """
        Scenario A: Selling BTC to buy a House
        """
        # 1. Selling BTC at time 0
        gross_amount = self.B0 * self.P0
        btc_selling_fee = gross_amount * self.F_BTC
        capital_gains = self.B0 * (self.P0 - self.P_basis)
        capital_gains_tax = capital_gains * self.tau
        net_proceeds = gross_amount - btc_selling_fee - capital_gains_tax
        
        # 2. Buy House
        house_cost = self.H0 * (1 + self.F_House)
        
        # Check if enough money to buy house
        can_afford = net_proceeds >= house_cost
        remaining_cash = net_proceeds - house_cost if can_afford else 0
        
        # 3. Holding cost over time (Present Value)
        holding_costs_pv = 0
        for t in range(self.T + 1):
            house_value_t = self.H0 * (1 + self.rh) ** t
            yearly_cost = self.f_House * house_value_t
            # Discount to present value
            holding_costs_pv += yearly_cost / ((1 + self.pi) ** t)
        
        # 4. Value at Time T
        final_house_value = self.H0 * (1 + self.rh) ** self.T
        
        # What if we kept the BTC instead?
        final_btc_value = self.B0 * self.P0 * (1 + self.rb) ** self.T
        
        # 5. Net Gain
        net_value = final_house_value - house_cost - holding_costs_pv + remaining_cash
        opportunity_cost = final_btc_value - gross_amount
        
        return {
            "scenario": "A - Sell BTC to Buy House",
            "can_afford_house": can_afford,
            "initial_btc_value": gross_amount,
            "net_proceeds_after_tax": net_proceeds,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "holding_costs_pv": holding_costs_pv,
            "final_house_value": final_house_value,
            "final_btc_value_if_held": final_btc_value,
            "net_value": net_value,
            "opportunity_cost": opportunity_cost,
            "total_return": (net_value / house_cost) * 100 if house_cost > 0 else 0,
            "btc_return_if_held": ((final_btc_value / gross_amount) - 1) * 100
        }
    
    def scenario_b_borrow_against_btc(self):
        """
        Scenario B: Borrowing Against BTC to buy a house
        """
        # 1. Loan amount and House cost
        loan_amount = self.LTV * self.B0 * self.P0
        house_cost = self.H0 * (1 + self.F_House)
        
        # Check if loan covers house cost
        can_afford = loan_amount >= house_cost
        remaining_cash = loan_amount - house_cost if can_afford else 0
        
        # 2. BTC expected value over time
        # Using deterministic growth formula for median path
        final_btc_price = self.P0 * (1 + self.rb) ** self.T
        final_btc_value = self.B0 * final_btc_price
        
        # 3. Liquidation threshold analysis
        loan_with_interest = loan_amount * (1 + self.i_st) ** self.T
        liquidation_threshold = 1.25 * self.LTV  # Typically liquidation at 125% of LTV
        
        # Simulate BTC price path for liquidation analysis
        time_points = np.linspace(0, self.T, 12 * self.T + 1)  # Monthly checks
        liquidation_occurred = False
        liquidation_time = None
        
        # Track values over time for charts
        btc_value_history = []
        loan_value_history = []
        ltv_history = []
        
        for t in time_points:
            # Price at time t (median path)
            price_t = self.P0 * (1 + self.rb) ** t
            
            # Check for liquidation
            loan_value_t = loan_amount * (1 + self.i_st) ** t
            btc_value_t = self.B0 * price_t
            current_ltv = loan_value_t / btc_value_t
            
            # Store values for charts
            btc_value_history.append(btc_value_t)
            loan_value_history.append(loan_value_t)
            ltv_history.append(current_ltv * 100)  # Convert to percentage
            
            if current_ltv > liquidation_threshold and not liquidation_occurred:
                liquidation_occurred = True
                liquidation_time = t
                break
        
        # 4. Future value calculation
        final_house_value = self.H0 * (1 + self.rh) ** self.T
        
        if liquidation_occurred:
            # If liquidation happened, we lose the BTC but keep the house
            net_value = final_house_value - loan_with_interest + remaining_cash
        else:
            # No liquidation
            net_value = final_house_value + final_btc_value - loan_with_interest + remaining_cash
        
        return {
            "scenario": "B - Borrow Against BTC",
            "can_afford_house": can_afford,
            "loan_amount": loan_amount,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "loan_with_interest": loan_with_interest,
            "final_btc_price": final_btc_price,
            "final_btc_value": final_btc_value,
            "final_house_value": final_house_value,
            "liquidation_occurred": liquidation_occurred,
            "liquidation_time": liquidation_time,
            "net_value": net_value,
            "total_return": (net_value / self.initial_btc_value) * 100,
            "time_points": time_points,
            "btc_value_history": btc_value_history,
            "loan_value_history": loan_value_history,
            "ltv_history": ltv_history
        }
    
    def scenario_c_btc_collateral(self):
        """
        Scenario C: BTC as Secondary Collateral (Self-paying Mortgage)
        """
        # Initial loan parameters
        loan_amount = self.H0 * (1 + self.F_House)
        debt_remaining = loan_amount
        house_value = self.H0
        
        # Track metrics over time
        time_points = np.linspace(0, self.T, self.T + 1)  # Yearly checks
        debt_history = [debt_remaining]
        house_value_history = [house_value]
        btc_value_history = [self.B0 * self.P0]
        collateral_ratio_history = [(house_value + self.B0 * self.P0) / debt_remaining]
        liquidation_occurred = False
        liquidation_time = None
        
        # Simulate over time (yearly steps)
        for t in range(1, self.T + 1):
            # Update house value
            house_value = self.H0 * (1 + self.rh) ** t
            house_value_history.append(house_value)
            
            # Update BTC value
            btc_price_t = self.P0 * (1 + self.rb) ** t
            btc_value_t = self.B0 * btc_price_t
            btc_value_history.append(btc_value_t)
            
            # Calculate debt with interest
            debt_with_interest = debt_remaining * (1 + self.i_st)
            
            # Check if BTC + house value can cover the debt
            total_collateral = house_value + btc_value_t
            required_collateral = debt_with_interest * 1.25  # 125% LTV requirement
            
            if total_collateral < required_collateral and not liquidation_occurred:
                liquidation_occurred = True
                liquidation_time = t
                # In case of liquidation, we lose the BTC
                excess_value = 0
            else:
                # Calculate excess collateral that can be used to pay down principal
                excess_value = max(0, total_collateral - required_collateral)
                
            # Apply excess to debt reduction
            principal_reduction = min(excess_value, debt_with_interest)
            
            # Update debt remaining
            debt_remaining = debt_with_interest - principal_reduction
            debt_history.append(debt_remaining)
            
            # Calculate collateral ratio
            if debt_remaining > 0:
                collateral_ratio = (house_value + btc_value_t) / debt_remaining
            else:
                collateral_ratio = float('inf')  # No debt
            collateral_ratio_history.append(collateral_ratio)
            
            # If debt is fully paid, break
            if debt_remaining <= 0:
                debt_remaining = 0
                # Fill the rest of the history with zeros
                for i in range(t+1, self.T + 1):
                    debt_history.append(0)
                break
        
        # Final values
        final_house_value = house_value_history[-1]
        final_btc_value = btc_value_history[-1] if not liquidation_occurred else 0
        final_debt = debt_history[-1]
        
        # Net value calculation
        net_value = final_house_value + final_btc_value - final_debt
        
        return {
            "scenario": "C - BTC as Secondary Collateral",
            "initial_loan": loan_amount,
            "final_house_value": final_house_value,
            "final_btc_value": final_btc_value,
            "final_debt": final_debt,
            "liquidation_occurred": liquidation_occurred,
            "liquidation_time": liquidation_time,
            "debt_paid_off": debt_remaining == 0,
            "debt_history": debt_history,
            "house_value_history": house_value_history,
            "btc_value_history": btc_value_history,
            "collateral_ratio_history": collateral_ratio_history,
            "time_points": time_points,
            "net_value": net_value,
            "total_return": (net_value / self.initial_btc_value) * 100
        }
    
    def simulate_scenarios(self, num_simulations=100):
        """
        Run Monte Carlo simulations for all scenarios to generate bull/bear/base cases
        """
        # Store results for each scenario
        scenario_a_results = []
        scenario_b_results = []
        scenario_c_results = []
        
        # Run simulations
        for _ in range(num_simulations):
            # Generate random BTC and house appreciation rates
            # This simulates bull/bear market conditions
            btc_rate = np.random.normal(self.rb, self.sigma_b)
            house_rate = np.random.normal(self.rh, self.rh * 0.2)  # 20% volatility of base rate
            
            # Create temporary model with these rates
            temp_model = BTCHousingModel(
                initial_btc=self.B0,
                current_btc_price=self.P0,
                btc_purchase_price=self.P_basis,
                btc_appreciation_rate=btc_rate,
                initial_house_price=self.H0,
                house_appreciation_rate=house_rate,
                capital_gains_tax=self.tau,
                loan_to_value_ratio=self.LTV,
                mortgage_rate=self.i_st,
                time_horizon=self.T,
                btc_volatility=self.sigma_b,
                inflation_rate=self.pi,
                btc_selling_fee=self.F_BTC,
                house_purchase_fee=self.F_House,
                annual_house_cost=self.f_House
            )
            
            # Run the three scenarios
            result_a = temp_model.scenario_a_sell_btc()
            result_b = temp_model.scenario_b_borrow_against_btc()
            result_c = temp_model.scenario_c_btc_collateral()
            
            # Store the net value results
            scenario_a_results.append(result_a["net_value"])
            scenario_b_results.append(result_b["net_value"])
            scenario_c_results.append(result_c["net_value"])
        
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
                "All Results": scenario_a_results
            },
            "Scenario B": {
                "Bear Case": scenario_b_cases[0],
                "Base Case": scenario_b_cases[1],
                "Bull Case": scenario_b_cases[2],
                "All Results": scenario_b_results
            },
            "Scenario C": {
                "Bear Case": scenario_c_cases[0],
                "Base Case": scenario_c_cases[1],
                "Bull Case": scenario_c_cases[2],
                "All Results": scenario_c_results
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
        current_btc_price=inputs["current_btc_price"],
        btc_purchase_price=inputs["btc_purchase_price"],
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
        annual_house_cost=inputs["annual_house_cost"]
    )
    
    # Run scenarios
    scenario_a = model.scenario_a_sell_btc()
    scenario_b = model.scenario_b_borrow_against_btc()
    scenario_c = model.scenario_c_btc_collateral()
    
    # Run simulations
    simulations = model.simulate_scenarios(num_simulations=500)
    
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
            if scenario["liquidation_occurred"] and scenario["liquidation_time"] is not None:
                st.write(f"**Liquidation Time:** Year {scenario['liquidation_time']:.1f}")
        
        if "debt_paid_off" in scenario:
            st.write(f"**Debt Paid Off:** {'Yes' if scenario['debt_paid_off'] else 'No'}")
        
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
            # Cap the ratio at 10 for better visualization
            capped_ratios = [min(ratio, 10) for ratio in scenario_c["collateral_ratio_history"]]
            
            fig_collateral = go.Figure()
            
            fig_collateral.add_trace(go.Scatter(
                x=scenario_c["time_points"],
                y=capped_ratios,
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
        recommendation["advice"] = "Borrowing against BTC has liquidation risk if BTC price falls significantly."
    elif best_scenario == "C":
        recommendation["advice"] = "Using BTC as collateral requires careful monitoring and may lead to complex tax situations."
    
    return recommendation

def main():
    st.title("Bitcoin Housing Strategy Analyzer")
    st.markdown("""
    Compare three strategies for using Bitcoin to purchase a house:
    - **Scenario A**: Sell BTC to buy a house
    - **Scenario B**: Borrow against BTC to buy a house
    - **Scenario C**: Use BTC as secondary collateral
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Inputs", "Results", "Charts", "Comparison"])
    
    with tab1:
        st.header("Input Parameters")
        
        # Use columns for more compact layout
        col1, col2 = st.columns(2)
        
        # Bitcoin Parameters
        with col1:
            st.subheader("Bitcoin Parameters")
            initial_btc = st.number_input("Initial BTC Holdings", 
                                         min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            
            current_btc_price = st.number_input("Current BTC Price ($)", 
                                         min_value=1000, max_value=1000000, value=50000, step=1000,
                                         help="The current market price of Bitcoin at the time of analysis")
            
            btc_purchase_price = st.number_input("BTC Purchase Price ($)", 
                                           min_value=1000, max_value=1000000, value=20000, step=1000,
                                           help="The price at which you originally acquired your Bitcoin (used for tax calculations)")
            
            btc_appreciation_rate = st.slider("BTC Annual Appreciation (%)", 
                                           min_value=-20.0, max_value=100.0, value=20.0, step=1.0) / 100
            
            btc_volatility = st.slider("BTC Volatility", 
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
        
        # Time Parameters
        st.subheader("Time Parameters")
        time_horizon = st.slider("Time Horizon (Years)", 
                               min_value=1, max_value=30, value=10, step=1)
        
        # Save/Load inputs
        st.subheader("Save/Load Configuration")
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("Save Current Inputs"):
                inputs = {
                    "initial_btc": initial_btc,
                    "current_btc_price": current_btc_price,
                    "btc_purchase_price": btc_purchase_price,
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
                    "annual_house_cost": annual_house_cost
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
            with st.spinner("Running analysis..."):
                # Get input values
                inputs = {
                    "initial_btc": initial_btc,
                    "current_btc_price": current_btc_price,
                    "btc_purchase_price": btc_purchase_price,
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
                    "annual_house_cost": annual_house_cost
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
                st.experimental_rerun()

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
                    "Yes" if st.session_state.scenario_b["liquidation_occurred"] else "No"
                ],
                "Scenario C": [
                    format_currency(st.session_state.scenario_c["net_value"]),
                    format_percent(st.session_state.scenario_c["total_return"]),
                    format_currency(st.session_state.scenario_c["final_house_value"]),
                    format_currency(st.session_state.scenario_c["final_btc_value"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Bear Case"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Base Case"]),
                    format_currency(st.session_state.simulations["Scenario C"]["Bull Case"]),
                    "Yes" if st.session_state.scenario_c["liquidation_occurred"] else "No"
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
                    0 if st.session_state.scenario_b["liquidation_occurred"] else 80
                ],
                "Scenario C": [
                    st.session_state.scenario_c["net_value"],
                    st.session_state.scenario_c["total_return"],
                    st.session_state.scenario_c["final_house_value"],
                    st.session_state.scenario_c["final_btc_value"],
                    st.session_state.simulations["Scenario C"]["Bear Case"],
                    st.session_state.simulations["Scenario C"]["Base Case"],
                    st.session_state.simulations["Scenario C"]["Bull Case"],
                    0 if st.session_state.scenario_c["liquidation_occurred"] else 60
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
    
    # Add footer
    st.markdown("---")
    st.markdown("Bitcoin Housing Strategy Analyzer - Helping you make informed decisions about Bitcoin and real estate")

if __name__ == "__main__":
    main()
