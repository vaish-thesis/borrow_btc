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
    Model for comparing 2 scenarios of using Bitcoin for house purchases:
    A) Sell BTC to buy house (opportunity cost analysis)
    B) Borrow against BTC to buy house with fixed 125% LTV
    """
    
    def __init__(self, 
                 initial_btc,              # B₀: Initial BTC holdings
                 initial_btc_price,        # P₀: BTC price at time t=0
                 btc_basis_price,          # Price at which BTC was acquired (for tax purposes)
                 btc_drift,                # μ: BTC annual drift
                 initial_house_price,      # H₀: Initial house price (USD)
                 house_appreciation_rate,  # rₕ: Rate of house appreciation
                 capital_gains_tax,        # τ: Capital gains tax rate
                 mortgage_rate,            # iₛₜ: Mortgage rate
                 time_horizon,             # T: Time horizon (years)
                 btc_volatility,           # σ: BTC volatility
                 inflation_rate,           # π: Inflation rate
                 btc_selling_fee,          # F_BTC: Fee for selling BTC (%)
                 house_purchase_fee,       # F_House: House purchase fee (%)
                 annual_house_cost,        # f_House: Annual recurring house cost (%)
                 num_simulations=1000,     # Number of Monte Carlo simulations
                 num_time_steps=365        # Number of time steps per year
                ):
        
        # Store parameters
        self.B0 = initial_btc
        self.P0 = initial_btc_price
        self.P_basis = btc_basis_price
        self.mu = btc_drift
        self.H0 = initial_house_price
        self.rh = house_appreciation_rate
        self.tau = capital_gains_tax
        self.LTV = 0.8  # Fixed at 80% (125% maintenance margin)
        self.i_st = mortgage_rate
        self.T = time_horizon
        self.sigma = btc_volatility
        self.pi = inflation_rate
        self.F_BTC = btc_selling_fee
        self.F_House = house_purchase_fee
        self.f_House = annual_house_cost
        self.num_simulations = num_simulations
        self.num_time_steps = num_time_steps
        
        # Derived values
        self.initial_btc_value = self.B0 * self.P0
        
    def _simulate_gbm_paths(self):
        """
        Generate price paths using Geometric Brownian Motion
        Returns a 2D array where each row is a price path simulation
        """
        dt = self.T / self.num_time_steps
        paths = np.zeros((self.num_simulations, self.num_time_steps + 1))
        paths[:, 0] = self.P0
        
        for t in range(1, self.num_time_steps + 1):
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            shock = self.sigma * np.sqrt(dt) * np.random.normal(0, 1, size=self.num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(drift + shock)
            
        return paths
    
    def scenario_a_sell_btc(self):
        """
        Scenario A: Selling BTC to buy a House
        Calculates opportunity cost of selling BTC
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
        
        # Run simulations for opportunity cost analysis
        btc_price_paths = self._simulate_gbm_paths()
        final_btc_prices = btc_price_paths[:, -1]
        final_btc_values = self.B0 * final_btc_prices
        
        # Calculate opportunity costs (what if we kept the BTC instead)
        opportunity_costs = final_btc_values - gross_amount
        
        # Calculate percentiles for opportunity cost
        percentiles = [10, 25, 50, 75, 90]
        opportunity_cost_percentiles = np.percentile(opportunity_costs, percentiles)
        
        # 5. Net Gain for housing approach
        net_value = final_house_value - house_cost - holding_costs_pv + remaining_cash
        
        return {
            "scenario": "A - Sell BTC to Buy House",
            "can_afford_house": can_afford,
            "initial_btc_value": gross_amount,
            "net_proceeds_after_tax": net_proceeds,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "holding_costs_pv": holding_costs_pv,
            "final_house_value": final_house_value,
            "btc_price_paths": btc_price_paths,
            "final_btc_values": final_btc_values,
            "opportunity_costs": opportunity_costs,
            "opportunity_cost_percentiles": {
                "p10": opportunity_cost_percentiles[0],
                "p25": opportunity_cost_percentiles[1],
                "p50": opportunity_cost_percentiles[2],
                "p75": opportunity_cost_percentiles[3],
                "p90": opportunity_cost_percentiles[4]
            },
            "net_value": net_value,
            "total_return": (net_value / house_cost) * 100 if house_cost > 0 else 0
        }
    
    def scenario_b_borrow_against_btc(self):
        """
        Scenario B: Borrowing Against BTC to buy a house
        Using 125% liquidation threshold (fixed)
        """
        # 1. Loan amount and House cost
        loan_amount = self.LTV * self.B0 * self.P0
        house_cost = self.H0 * (1 + self.F_House)
        
        # Check if loan covers house cost
        can_afford = loan_amount >= house_cost
        remaining_cash = loan_amount - house_cost if can_afford else 0
        
        # 2. Simulate BTC price paths
        btc_price_paths = self._simulate_gbm_paths()
        
        # 3. Analyze liquidation events across all simulations
        # Typically liquidation at 125% of LTV (or LTV at 80%)
        liquidation_threshold = 1.25 * self.LTV  
        
        # Track liquidations across simulations
        liquidation_occurred = np.zeros(self.num_simulations, dtype=bool)
        liquidation_times = np.full(self.num_simulations, np.nan)
        
        # Calculate loan value over time (assuming compounding)
        time_points = np.linspace(0, self.T, self.num_time_steps + 1)
        # Create a loan values array with same shape as btc_price_paths
        loan_values = np.zeros_like(btc_price_paths)
        for i in range(self.num_simulations):
            loan_values[i] = loan_amount * (1 + self.i_st) ** time_points
        
        # Calculate LTV ratios for each path at each time point
        btc_values = self.B0 * btc_price_paths
        ltv_ratios = np.zeros_like(btc_price_paths)
        for i in range(self.num_simulations):
            ltv_ratios[i] = loan_values[i] / btc_values[i]
            
            # Check for liquidation
            liquidation_indices = np.where(ltv_ratios[i] > liquidation_threshold)[0]
            if len(liquidation_indices) > 0:
                liquidation_occurred[i] = True
                liquidation_times[i] = time_points[liquidation_indices[0]]
        
        # Calculate final values for each simulation
        final_house_value = self.H0 * (1 + self.rh) ** self.T
        loan_with_interest = loan_amount * (1 + self.i_st) ** self.T
        
        # Net value calculation for each simulation
        final_btc_values = np.zeros(self.num_simulations)
        net_values = np.zeros(self.num_simulations)
        
        for i in range(self.num_simulations):
            if liquidation_occurred[i]:
                # If liquidation happened, we lose the BTC but keep the house
                final_btc_values[i] = 0
                net_values[i] = final_house_value - loan_with_interest + remaining_cash
            else:
                # No liquidation
                final_btc_values[i] = self.B0 * btc_price_paths[i, -1]
                net_values[i] = final_house_value + final_btc_values[i] - loan_with_interest + remaining_cash
        
        # Calculate percentiles for net values
        percentiles = [10, 25, 50, 75, 90]
        net_value_percentiles = np.percentile(net_values, percentiles)
        
        return {
            "scenario": "B - Borrow Against BTC (125% Liquidation Threshold)",
            "can_afford_house": can_afford,
            "loan_amount": loan_amount,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "loan_with_interest": loan_with_interest,
            "final_house_value": final_house_value,
            "btc_price_paths": btc_price_paths,
            "liquidation_occurred": liquidation_occurred,
            "liquidation_times": liquidation_times,
            "liquidation_probability": np.mean(liquidation_occurred) * 100,
            "median_liquidation_time": np.nanmedian(liquidation_times) if np.any(liquidation_occurred) else None,
            "ltv_ratios": ltv_ratios,
            "time_points": time_points,
            "loan_values": loan_values,
            "btc_values": btc_values,
            "net_values": net_values,
            "final_btc_values": final_btc_values,
            "net_value_percentiles": {
                "p10": net_value_percentiles[0],
                "p25": net_value_percentiles[1],
                "p50": net_value_percentiles[2],
                "p75": net_value_percentiles[3],
                "p90": net_value_percentiles[4]
            },
            "median_net_value": net_value_percentiles[2],
            "total_return": (net_value_percentiles[2] / self.initial_btc_value) * 100
        }
    
    def simulate_scenarios(self):
        """
        Run both scenarios and generate comparative analytics
        """
        # Run both scenarios
        scenario_a = self.scenario_a_sell_btc()
        scenario_b = self.scenario_b_borrow_against_btc()
        
        # Generate comparative analytics
        comparison = {
            "better_median_outcome": "A" if scenario_a["net_value"] > scenario_b["median_net_value"] else "B",
            "opportunity_cost_median": scenario_a["opportunity_cost_percentiles"]["p50"],
            "opportunity_cost_risk": scenario_a["opportunity_cost_percentiles"]["p90"] - scenario_a["opportunity_cost_percentiles"]["p10"],
            "liquidation_probability": scenario_b["liquidation_probability"],
            "risk_difference": {
                "a_downside": scenario_a["net_value"] - scenario_a["opportunity_cost_percentiles"]["p10"],
                "b_downside": scenario_b["net_value_percentiles"]["p50"] - scenario_b["net_value_percentiles"]["p10"]
            }
        }
        
        return {
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "comparison": comparison
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
        btc_drift=inputs["btc_drift"],
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
        num_simulations=inputs["num_simulations"]
    )
    
    # Run scenarios
    simulation_results = model.simulate_scenarios()
    
    return model, simulation_results

def display_scenario_a_metrics(scenario, col):
    """Display metrics for scenario A in a Streamlit column"""
    col.subheader(scenario["scenario"])
    
    # Create a metrics section
    col.metric("Net Value", format_currency(scenario["net_value"]))
    col.metric("Total Return", format_percent(scenario["total_return"]))
    
    # Create an expander for more details
    with col.expander("Detailed Metrics"):
        st.write(f"**Can Afford House:** {'Yes' if scenario['can_afford_house'] else 'No'}")
        st.write(f"**Final House Value:** {format_currency(scenario['final_house_value'])}")
        st.write(f"**Initial BTC Value:** {format_currency(scenario['initial_btc_value'])}")
        st.write(f"**Net Proceeds After Tax:** {format_currency(scenario['net_proceeds_after_tax'])}")
        st.write(f"**House Cost:** {format_currency(scenario['house_cost'])}")
        st.write(f"**Remaining Cash:** {format_currency(scenario['remaining_cash'])}")
    
    # Display opportunity cost analysis
    with col.expander("Opportunity Cost Analysis"):
        st.write("### Opportunity Cost Percentiles")
        st.write(f"**10th Percentile:** {format_currency(scenario['opportunity_cost_percentiles']['p10'])}")
        st.write(f"**25th Percentile:** {format_currency(scenario['opportunity_cost_percentiles']['p25'])}")
        st.write(f"**Median (50th):** {format_currency(scenario['opportunity_cost_percentiles']['p50'])}")
        st.write(f"**75th Percentile:** {format_currency(scenario['opportunity_cost_percentiles']['p75'])}")
        st.write(f"**90th Percentile:** {format_currency(scenario['opportunity_cost_percentiles']['p90'])}")
        
        # Histogram of opportunity costs
        fig = px.histogram(
            x=scenario["opportunity_costs"],
            nbins=50,
            labels={"x": "Opportunity Cost ($)"},
            title="Distribution of Opportunity Costs"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_scenario_b_metrics(scenario, col):
    """Display metrics for scenario B in a Streamlit column"""
    col.subheader(scenario["scenario"])
    
    # Create a metrics section
    col.metric("Median Net Value", format_currency(scenario["median_net_value"]))
    col.metric("Total Return", format_percent(scenario["total_return"]))
    col.metric("Liquidation Probability", format_percent(scenario["liquidation_probability"]))
    
    # Create an expander for more details
    with col.expander("Detailed Metrics"):
        st.write(f"**Can Afford House:** {'Yes' if scenario['can_afford_house'] else 'No'}")
        st.write(f"**Loan Amount:** {format_currency(scenario['loan_amount'])}")
        st.write(f"**House Cost:** {format_currency(scenario['house_cost'])}")
        st.write(f"**Final House Value:** {format_currency(scenario['final_house_value'])}")
        st.write(f"**Remaining Cash:** {format_currency(scenario['remaining_cash'])}")
        st.write(f"**Loan with Interest:** {format_currency(scenario['loan_with_interest'])}")
        if scenario["median_liquidation_time"] is not None:
            st.write(f"**Median Liquidation Time:** Year {scenario['median_liquidation_time']:.2f}")
    
    # Display outcome percentiles
    with col.expander("Outcome Percentiles"):
        st.write("### Net Value Percentiles")
        st.write(f"**10th Percentile:** {format_currency(scenario['net_value_percentiles']['p10'])}")
        st.write(f"**25th Percentile:** {format_currency(scenario['net_value_percentiles']['p25'])}")
        st.write(f"**Median (50th):** {format_currency(scenario['net_value_percentiles']['p50'])}")
        st.write(f"**75th Percentile:** {format_currency(scenario['net_value_percentiles']['p75'])}")
        st.write(f"**90th Percentile:** {format_currency(scenario['net_value_percentiles']['p90'])}")
        
        # Histogram of net values
        fig = px.histogram(
            x=scenario["net_values"],
            nbins=50,
            labels={"x": "Net Value ($)"},
            title="Distribution of Net Values"
        )
        st.plotly_chart(fig, use_container_width=True)

def create_price_paths_chart(scenario_a, scenario_b):
    """Create a chart showing BTC price paths from simulations"""
    # Sample a subset of price paths to avoid clutter
    num_paths_to_show = 100
    path_indices = np.random.choice(len(scenario_a["btc_price_paths"]), num_paths_to_show, replace=False)
    
    # Create a figure for the price paths
    fig = go.Figure()
    
    # Time points for the x-axis
    time_points = np.linspace(0, scenario_a["btc_price_paths"].shape[1]-1, scenario_a["btc_price_paths"].shape[1]) / (scenario_a["btc_price_paths"].shape[1]-1)
    
    # Add sample price paths
    for i in path_indices:
        fig.add_trace(go.Scatter(
            x=time_points,
            y=scenario_a["btc_price_paths"][i],
            mode='lines',
            line=dict(color='rgba(100, 149, 237, 0.1)'),  # Light blue with transparency
            showlegend=False
        ))
    
    # Add median path
    median_path = np.median(scenario_a["btc_price_paths"], axis=0)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=median_path,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Median Price Path'
    ))
    
    # Add 10th and 90th percentile paths
    p10_path = np.percentile(scenario_a["btc_price_paths"], 10, axis=0)
    p90_path = np.percentile(scenario_a["btc_price_paths"], 90, axis=0)
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=p10_path,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='10th Percentile'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=p90_path,
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='90th Percentile'
    ))
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price Simulations (Monte Carlo)',
        xaxis_title='Time (Normalized)',
        yaxis_title='BTC Price ($)',
        yaxis=dict(tickformat='$,'),
        height=500
    )
    
    return fig

def create_liquidation_chart(scenario_b):
    """Create a chart showing liquidation events from scenario B"""
    # Extract data
    time_points = scenario_b["time_points"]
    liquidation_occurred = scenario_b["liquidation_occurred"]
    liquidation_times = scenario_b["liquidation_times"]
    
    # Count liquidations by time period
    hist_data, bin_edges = np.histogram(
        liquidation_times[~np.isnan(liquidation_times)], 
        bins=20, 
        range=(0, max(time_points))
    )
    
    # Create histogram figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_edges[:-1],
        y=hist_data,
        marker_color='firebrick',
        name='Liquidation Events'
    ))
    
    # Add cumulative line
    cum_hist = np.cumsum(hist_data) / sum(hist_data) * 100
    
    fig.add_trace(go.Scatter(
        x=bin_edges[:-1],
        y=cum_hist,
        mode='lines+markers',
        marker=dict(size=8, color='navy'),
        line=dict(width=3),
        name='Cumulative %',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Liquidation Events Over Time (Overall Probability: {np.mean(liquidation_occurred)*100:.2f}%)',
        xaxis_title='Time (Years)',
        yaxis_title='Number of Liquidations',
        yaxis2=dict(
            title='Cumulative Percentage',
            titlefont=dict(color='navy'),
            tickfont=dict(color='navy'),
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=500
    )
    
    return fig

def create_outcome_comparison_chart(scenario_a, scenario_b):
    """Create a chart comparing outcome distributions"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add histograms as traces
    fig.add_trace(
        go.Histogram(
            x=scenario_a["opportunity_costs"],
            nbinsx=30,
            name="Scenario A - Opportunity Cost",
            marker_color='rgba(100, 149, 237, 0.7)'  # Light blue
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Histogram(
            x=scenario_b["net_values"],
            nbinsx=30,
            name="Scenario B - Net Value",
            marker_color='rgba(76, 175, 80, 0.7)'  # Light green
        ),
        secondary_y=True
    )
    
    # Add vertical lines for median values
    fig.add_vline(
        x=scenario_a["opportunity_cost_percentiles"]["p50"],
        line_dash="dash",
        line_color="blue",
        annotation_text="Median Opportunity Cost",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=scenario_b["net_value_percentiles"]["p50"],
        line_dash="dash", 
        line_color="green",
        annotation_text="Median Net Value",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title="Comparison of Outcome Distributions",
        xaxis_title="Value ($)",
        yaxis_title="Frequency (Scenario A)",
        yaxis2_title="Frequency (Scenario B)",
        barmode='overlay',
        height=500
    )
    
    fig.update_xaxes(tickformat='$,')
    
    return fig

def create_scenario_comparison_table(scenario_a, scenario_b):
    """Create a comparison table between scenarios A and B"""
    comparison_data = {
        "Metric": [
            "Median Outcome",
            "10th Percentile Outcome",
            "90th Percentile Outcome",
            "Probability of Liquidation",
            "Downside Risk",
            "Upside Potential",
            "Initial Investment",
            "Final House Value"
        ],
        "Scenario A": [
            format_currency(scenario_a["net_value"]),
            format_currency(scenario_a["net_value"] - scenario_a["opportunity_cost_percentiles"]["p90"]),
            format_currency(scenario_a["net_value"] - scenario_a["opportunity_cost_percentiles"]["p10"]),
            "0%",
            format_currency(scenario_a["opportunity_cost_percentiles"]["p90"]),
            format_currency(scenario_a["opportunity_cost_percentiles"]["p10"]),
            format_currency(scenario_a["house_cost"]),
            format_currency(scenario_a["final_house_value"])
        ],
        "Scenario B": [
            format_currency(scenario_b["net_value_percentiles"]["p50"]),
            format_currency(scenario_b["net_value_percentiles"]["p10"]),
            format_currency(scenario_b["net_value_percentiles"]["p90"]),
            format_percent(scenario_b["liquidation_probability"]),
            format_currency(scenario_b["net_value_percentiles"]["p50"] - scenario_b["net_value_percentiles"]["p10"]),
            format_currency(scenario_b["net_value_percentiles"]["p90"] - scenario_b["net_value_percentiles"]["p50"]),
            format_currency(scenario_b["loan_amount"]),
            format_currency(scenario_b["final_house_value"])
        ]
    }
    
    return pd.DataFrame(comparison_data)

def get_recommendation(scenario_a, scenario_b):
    """Get a recommendation based on the analysis"""
    # Determine best scenario based on median outcome
    a_outcome = scenario_a["net_value"]
    b_outcome = scenario_b["net_value_percentiles"]["p50"]
    
    best_median = "A" if a_outcome > b_outcome else "B"
    
    # Calculate downside risk
    a_downside = scenario_a["opportunity_cost_percentiles"]["p90"]  # High opportunity cost is bad
    b_downside = scenario_b["net_value_percentiles"]["p50"] - scenario_b["net_value_percentiles"]["p10"]
    
    lowest_risk = "A" if a_downside < b_downside else "B"
    
    # Calculate upside potential
    a_upside = scenario_a["opportunity_cost_percentiles"]["p10"]  # Low opportunity cost is good
    b_upside = scenario_b["net_value_percentiles"]["p90"] - scenario_b["net_value_percentiles"]["p50"]
    
    highest_upside = "A" if a_upside < 0 and abs(a_upside) > b_upside else "B"
    
    # Generate recommendation
    recommendation = {
        "best_median": best_median,
        "lowest_risk": lowest_risk,
        "highest_upside": highest_upside,
        "liquidation_probability": scenario_b["liquidation_probability"]
    }
    
    # Add some nuance
    if best_median == "A" and lowest_risk == "A":
        recommendation["advice"] = "Selling BTC provides better median outcome with lower risk, but may miss future BTC appreciation."
    elif best_median == "B" and highest_upside == "B":
        recommendation["advice"] = f"Borrowing against BTC provides better median outcome and upside, but has {scenario_b['liquidation_probability']:.1f}% liquidation risk."
    elif best_median == "A" and highest_upside == "B":
        recommendation["advice"] = "Selling BTC provides better median outcome, but borrowing has higher upside potential with additional risk."
    else:
        recommendation["advice"] = "Consider your risk tolerance and Bitcoin price outlook when choosing between selling and borrowing."
    
    return recommendation

def main():
    st.title("Bitcoin Housing Strategy Analyzer")
    st.markdown("""
    Compare two strategies for using Bitcoin to purchase a house:
    - **Scenario A**: Sell BTC to buy a house (opportunity cost analysis)
    - **Scenario B**: Borrow against BTC to buy a house (fixed 125% liquidation threshold)
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
            
            initial_btc_price = st.number_input("Current BTC Price ($)", 
                                             min_value=1000, max_value=1000000, value=50000, step=1000)
            
            btc_basis_price = st.number_input("BTC Basis Price ($)", 
                                           min_value=1000, max_value=1000000, value=20000, step=1000)
            
            btc_drift = st.slider("BTC Annual Drift (%)", 
                                   min_value=-20.0, max_value=100.0, value=20.0, step=1.0) / 100
            
            btc_volatility = st.slider("BTC Volatility", 
                                     min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            
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
        
        with col4:
            mortgage_rate = st.slider("Mortgage Rate (%)", 
                                    min_value=1.0, max_value=15.0, value=5.0, step=0.1) / 100
            
            inflation_rate = st.slider("Inflation Rate (%)", 
                                     min_value=0.0, max_value=20.0, value=3.0, step=0.1) / 100
        
        # Time Parameters and Simulation Settings
        st.subheader("Time and Simulation Parameters")
        col5, col6 = st.columns(2)
        
        with col5:
            time_horizon = st.slider("Time Horizon (Years)", 
                                   min_value=1, max_value=30, value=10, step=1)
        
        with col6:
            num_simulations = st.slider("Number of Monte Carlo Simulations", 
                                      min_value=100, max_value=10000, value=1000, step=100)
        
        # Save/Load inputs
        st.subheader("Save/Load Configuration")
        col7, col8 = st.columns(2)
        
        with col7:
            if st.button("Save Current Inputs"):
                inputs = {
                    "initial_btc": initial_btc,
                    "initial_btc_price": initial_btc_price,
                    "btc_basis_price": btc_basis_price,
                    "btc_drift": btc_drift,
                    "initial_house_price": initial_house_price,
                    "house_appreciation_rate": house_appreciation_rate,
                    "capital_gains_tax": capital_gains_tax,
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
        
        with col8:
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
                    "btc_drift": btc_drift,
                    "initial_house_price": initial_house_price,
                    "house_appreciation_rate": house_appreciation_rate,
                    "capital_gains_tax": capital_gains_tax,
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
                model, simulation_results = run_analysis(inputs)
                
                # Store results in session state
                st.session_state.model = model
                st.session_state.simulation_results = simulation_results
                
                # Switch to results tab
                st.rerun()

    with tab2:
        if 'simulation_results' in st.session_state:
            st.header("Analysis Results")
            
            # Extract scenarios from simulation results
            scenario_a = st.session_state.simulation_results["scenario_a"]
            scenario_b = st.session_state.simulation_results["scenario_b"]
            
            # Display metrics for each scenario in columns
            col1, col2 = st.columns(2)
            
            display_scenario_a_metrics(scenario_a, col1)
            display_scenario_b_metrics(scenario_b, col2)
            
            # Display recommendation
            st.subheader("Recommendation")
            recommendation = get_recommendation(scenario_a, scenario_b)
            
            # Use columns for recommendation
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"**Best Median Outcome:** Scenario {recommendation['best_median']}")
                st.write(f"**Lowest Risk:** Scenario {recommendation['lowest_risk']}")
                st.write(f"**Highest Upside:** Scenario {recommendation['highest_upside']}")
                st.write(f"**Advice:** {recommendation['advice']}")
            
            with col2:
                # Create a gauge for liquidation probability
                liquidation_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = recommendation["liquidation_probability"],
                    title = {'text': "Liquidation Risk"},
                    gauge = {
                        'axis': {'range': [0, 100], 'ticksuffix': "%"},
                        'bar': {'color': "darkred"},
                        'steps' : [
                            {'range': [0, 20], 'color': "green"},
                            {'range': [20, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "red"}
                        ],
                    }
                ))
                
                liquidation_gauge.update_layout(height=200, width=250, margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(liquidation_gauge, use_container_width=True)
        else:
            st.info("Run the analysis in the Inputs tab to see results")
    
    with tab3:
        if 'simulation_results' in st.session_state:
            st.header("Visualization Charts")
            
            # Extract scenarios from simulation results
            scenario_a = st.session_state.simulation_results["scenario_a"]
            scenario_b = st.session_state.simulation_results["scenario_b"]
            
            # Create BTC price paths chart
            price_paths_chart = create_price_paths_chart(scenario_a, scenario_b)
            st.plotly_chart(price_paths_chart, use_container_width=True)
            
            # Create liquidation chart
            liquidation_chart = create_liquidation_chart(scenario_b)
            st.plotly_chart(liquidation_chart, use_container_width=True)
            
            # Create outcome comparison chart
            outcome_comparison_chart = create_outcome_comparison_chart(scenario_a, scenario_b)
            st.plotly_chart(outcome_comparison_chart, use_container_width=True)
            
            # Create sample LTV ratio paths chart
            st.subheader("LTV Ratio Paths for Sample Simulations")
            num_paths_to_show = 20
            path_indices = np.random.choice(len(scenario_b["ltv_ratios"]), num_paths_to_show, replace=False)
            
            ltv_fig = go.Figure()
            
            # Add sample LTV paths
            for i in path_indices:
                ltv_fig.add_trace(go.Scatter(
                    x=scenario_b["time_points"],
                    y=scenario_b["ltv_ratios"][i] * 100,  # Convert to percentage
                    mode='lines',
                    line=dict(color='rgba(76, 175, 80, 0.3)'),  # Light green with transparency
                    showlegend=False
                ))
            
            # Add median LTV path
            median_ltv_path = np.median(scenario_b["ltv_ratios"], axis=0) * 100
            ltv_fig.add_trace(go.Scatter(
                x=scenario_b["time_points"],
                y=median_ltv_path,
                mode='lines',
                line=dict(color='green', width=3),
                name='Median LTV Path'
            ))
            
            # Add liquidation threshold line
            ltv_fig.add_trace(go.Scatter(
                x=scenario_b["time_points"],
                y=[125] * len(scenario_b["time_points"]),
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Liquidation Threshold (125%)'
            ))
            
            # Update layout
            ltv_fig.update_layout(
                title='Loan-to-Value Ratio Simulations (125% Liquidation Threshold)',
                xaxis_title='Time (Years)',
                yaxis_title='LTV Ratio (%)',
                height=500
            )
            
            st.plotly_chart(ltv_fig, use_container_width=True)
            
        else:
            st.info("Run the analysis in the Inputs tab to see charts")
    
    with tab4:
        if 'simulation_results' in st.session_state:
            st.header("Scenario Comparison")
            
            # Extract scenarios from simulation results
            scenario_a = st.session_state.simulation_results["scenario_a"]
            scenario_b = st.session_state.simulation_results["scenario_b"]
            
            # Create comparison table
            comparison_table = create_scenario_comparison_table(scenario_a, scenario_b)
            st.dataframe(comparison_table, use_container_width=True)
            
            # Create radar chart for comparison
            st.subheader("Strategy Comparison Radar Chart")
            
            # Prepare data for radar chart
            radar_categories = ['Median Outcome', 'Downside Protection', 'Upside Potential', 
                             'Avoids Liquidation', 'BTC Retention', 'House Ownership']
            
            # Normalize values for radar chart (0-100 scale)
            a_median = scenario_a["net_value"]
            b_median = scenario_b["net_value_percentiles"]["p50"]
            better_median = max(a_median, b_median)
            
            a_downside = -min(0, scenario_a["opportunity_cost_percentiles"]["p90"])  # Higher is better
            b_downside = scenario_b["net_value_percentiles"]["p10"]
            better_downside = max(a_downside, b_downside)
            
            a_upside = -min(0, scenario_a["opportunity_cost_percentiles"]["p10"])  # Higher is better
            b_upside = scenario_b["net_value_percentiles"]["p90"]
            better_upside = max(a_upside, b_upside)
            
            radar_data = {
                "category": radar_categories,
                "Scenario A": [
                    a_median / better_median * 100 if better_median > 0 else 0,  # Median Outcome
                    a_downside / better_downside * 100 if better_downside > 0 else 0,  # Downside Protection
                    a_upside / better_upside * 100 if better_upside > 0 else 0,  # Upside Potential
                    100,  # Avoids Liquidation
                    0,    # BTC Retention
                    100   # House Ownership
                ],
                "Scenario B": [
                    b_median / better_median * 100 if better_median > 0 else 0,  # Median Outcome
                    b_downside / better_downside * 100 if better_downside > 0 else 0,  # Downside Protection
                    b_upside / better_upside * 100 if better_upside > 0 else 0,  # Upside Potential
                    100 - scenario_b["liquidation_probability"],  # Avoids Liquidation
                    100 * (1 - scenario_b["liquidation_probability"]/100),  # BTC Retention
                    100   # House Ownership
                ]
            }
            
            # Create radar chart
            radar_fig = go.Figure()
            
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data["Scenario A"],
                theta=radar_data["category"],
                fill='toself',
                name='Scenario A',
                line_color='blue'
            ))
            
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data["Scenario B"],
                theta=radar_data["category"],
                fill='toself',
                name='Scenario B',
                line_color='green'
            ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Strategy Comparison Radar",
                height=600
            )
            
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Create a metrics breakdown
            st.subheader("Detailed Metrics Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Scenario A: Sell BTC to Buy House")
                st.write(f"**Median Outcome:** {format_currency(scenario_a['net_value'])}")
                st.write(f"**House Cost:** {format_currency(scenario_a['house_cost'])}")
                st.write(f"**Final House Value:** {format_currency(scenario_a['final_house_value'])}")
                st.write(f"**Median Opportunity Cost:** {format_currency(scenario_a['opportunity_cost_percentiles']['p50'])}")
                st.write(f"**Maximum (90%) Opportunity Cost:** {format_currency(scenario_a['opportunity_cost_percentiles']['p90'])}")
                st.write(f"**Minimum (10%) Opportunity Cost:** {format_currency(scenario_a['opportunity_cost_percentiles']['p10'])}")
            
            with col2:
                st.write("### Scenario B: Borrow Against BTC")
                st.write(f"**Median Outcome:** {format_currency(scenario_b['net_value_percentiles']['p50'])}")
                st.write(f"**Loan Amount:** {format_currency(scenario_b['loan_amount'])}")
                st.write(f"**Final House Value:** {format_currency(scenario_b['final_house_value'])}")
                st.write(f"**Loan with Interest:** {format_currency(scenario_b['loan_with_interest'])}")
                st.write(f"**Liquidation Probability:** {format_percent(scenario_b['liquidation_probability'])}")
                st.write(f"**10% Worst Case:** {format_currency(scenario_b['net_value_percentiles']['p10'])}")
                st.write(f"**90% Best Case:** {format_currency(scenario_b['net_value_percentiles']['p90'])}")
            
        else:
            st.info("Run the analysis in the Inputs tab to see comparison")
    
    # Add footer
    st.markdown("---")
    st.markdown("Bitcoin Housing Strategy Analyzer - Compare selling BTC vs. using BTC as collateral for buying a house")

if __name__ == "__main__":
    main()
