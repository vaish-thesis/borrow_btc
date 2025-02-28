import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the core model class
class BTCHousingModel:
    def __init__(self, initial_btc, initial_btc_price, btc_basis_price, initial_house_price,
                 btc_appreciation_rate, house_appreciation_rate, inflation_rate,
                 btc_volatility, btc_selling_fee, house_buying_fee, house_holding_cost,
                 tax_rate, loan_interest_rate, ltv_ratio, time_horizon):
        """Initialize the model with user inputs."""
        self.B0 = initial_btc
        self.P0 = initial_btc_price
        self.P_basis = btc_basis_price
        self.H0 = initial_house_price
        self.rb = btc_appreciation_rate
        self.rh = house_appreciation_rate
        self.pi = inflation_rate
        self.sigma_b = btc_volatility
        self.F_BTC = btc_selling_fee
        self.F_House = house_buying_fee
        self.f_House = house_holding_cost
        self.tau = tax_rate
        self.i_st = loan_interest_rate
        self.LTV = ltv_ratio
        self.T = time_horizon

    def generate_price_path(self, btc_rate, num_years, num_paths=1):
        """Generate BTC price paths using geometric Brownian motion."""
        paths = []
        for _ in range(num_paths):
            prices = [self.P0]
            for t in range(1, num_years + 1):
                Z = np.random.normal(0, 1)
                drift = (btc_rate - 0.5 * self.sigma_b ** 2)
                diffusion = self.sigma_b * Z
                price_t = prices[-1] * np.exp(drift + diffusion)
                prices.append(price_t)
            paths.append(prices)
        return paths

    def scenario_a_sell_btc(self):
        """Scenario A: Sell BTC to buy house outright."""
        gross_amount = self.B0 * self.P0
        btc_selling_fee = gross_amount * self.F_BTC
        capital_gains = self.B0 * (self.P0 - self.P_basis)
        capital_gains_tax = max(capital_gains * self.tau, 0)
        net_proceeds = gross_amount - btc_selling_fee - capital_gains_tax
        house_cost = self.H0 * (1 + self.F_House)
        can_afford = net_proceeds >= house_cost
        remaining_cash = net_proceeds - house_cost if can_afford else 0
        total_holding_costs = sum(self.f_House * self.H0 * (1 + self.rh) ** t for t in range(self.T + 1))
        holding_costs_pv = sum((self.f_House * self.H0 * (1 + self.rh) ** t) / ((1 + self.pi) ** t) for t in range(self.T + 1))
        final_house_value = self.H0 * (1 + self.rh) ** self.T if can_afford else 0
        final_btc_value = self.B0 * self.P0 * (1 + self.rb) ** self.T
        net_value = (final_house_value - house_cost - holding_costs_pv + remaining_cash) if can_afford else -house_cost
        return {
            "scenario": "A - Sell BTC to Buy House",
            "can_afford_house": can_afford,
            "net_proceeds_after_tax": net_proceeds,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "total_holding_costs": total_holding_costs,
            "holding_costs_pv": holding_costs_pv,
            "final_house_value": final_house_value,
            "final_btc_value_if_held": final_btc_value,
            "net_value": net_value,
            "opportunity_cost": final_btc_value - gross_amount
        }

    def scenario_b_borrow_against_btc(self, btc_rate=None, house_rate=None, num_paths=100):
        """Scenario B: Borrow against BTC to fund house purchase."""
        btc_rate = btc_rate if btc_rate is not None else self.rb
        house_rate = house_rate if house_rate is not None else self.rh
        loan_amount = self.LTV * self.B0 * self.P0
        house_cost = self.H0 * (1 + self.F_House)
        can_afford = loan_amount >= house_cost
        remaining_cash = max(loan_amount - house_cost, 0) if can_afford else 0
        btc_price_paths = self.generate_price_path(btc_rate, self.T, num_paths)
        net_values = []
        liquidation_occurred_list = []

        for path in btc_price_paths:
            liquidation_occurred = False
            btc_value_t = self.B0 * path[0]
            loan_value_t = loan_amount if can_afford else 0
            for t in range(1, self.T + 1):
                btc_value_t = self.B0 * path[t]
                loan_value_t *= (1 + self.i_st)
                current_ltv = loan_value_t / btc_value_t if btc_value_t > 0 else float('inf')
                if current_ltv > 1.25 * self.LTV and not liquidation_occurred:
                    liquidation_occurred = True
                    break
            final_house_value = self.H0 * (1 + house_rate) ** self.T if can_afford else 0
            net_value = (final_house_value - loan_value_t + remaining_cash) if liquidation_occurred else \
                        (final_house_value + btc_value_t - loan_value_t + remaining_cash)
            net_values.append(net_value)
            liquidation_occurred_list.append(liquidation_occurred)

        avg_net_value = np.mean(net_values)
        liquidation_probability = np.mean(liquidation_occurred_list) * 100
        return {
            "scenario": "B - Borrow Against BTC",
            "can_afford_house": can_afford,
            "loan_amount": loan_amount,
            "house_cost": house_cost,
            "remaining_cash": remaining_cash,
            "avg_net_value": avg_net_value,
            "liquidation_probability": liquidation_probability,
            "net_values": net_values,
            "liquidation_occurred_list": liquidation_occurred_list
        }

    def scenario_c_self_paying_mortgage(self, btc_rate=None, house_rate=None, num_paths=100):
        """Scenario C: Self-Paying Mortgage via Bitcoin Gains"""
        btc_rate = btc_rate if btc_rate is not None else self.rb
        house_rate = house_rate if house_rate is not None else self.rh
        initial_loan = self.H0 * (1 + self.F_House)
        btc_price_paths = self.generate_price_path(btc_rate, self.T, num_paths)
        net_values = []
        liquidation_occurred_list = []

        for path in btc_price_paths:
            debt_remaining = initial_loan
            house_value = self.H0
            liquidation_occurred = False
            for t in range(1, self.T + 1):
                house_value = self.H0 * (1 + house_rate) ** t
                btc_value = self.B0 * path[t]
                debt_with_interest = debt_remaining * (1 + self.i_st)
                total_collateral = house_value + btc_value
                required_collateral = debt_with_interest * 1.25
                if total_collateral < required_collateral:
                    liquidation_occurred = True
                    net_value = house_value - debt_with_interest
                    break
                else:
                    excess_value = total_collateral - required_collateral
                    principal_reduction = min(excess_value, debt_with_interest)
                    debt_remaining = debt_with_interest - principal_reduction
                    if debt_remaining <= 0:
                        debt_remaining = 0
                        net_value = house_value + btc_value
                        break
            else:
                net_value = house_value + btc_value - debt_remaining
            net_values.append(net_value)
            liquidation_occurred_list.append(liquidation_occurred)

        avg_net_value = np.mean(net_values)
        liquidation_probability = np.mean(liquidation_occurred_list) * 100
        return {
            "scenario": "C - Self-Paying Mortgage via Bitcoin Gains",
            "initial_loan": initial_loan,
            "avg_net_value": avg_net_value,
            "liquidation_probability": liquidation_probability,
            "net_values": net_values,
            "liquidation_occurred_list": liquidation_occurred_list
        }

    def simulate_scenarios(self, num_simulations=500):
        """Run Monte Carlo simulations for all scenarios with debug outputs."""
        scenario_a_net_values = []
        scenario_b_net_values = []
        scenario_c_net_values = []
        scenario_b_liquidation = []
        scenario_c_liquidation = []

        for sim in range(num_simulations):
            btc_rate = np.random.normal(self.rb, self.sigma_b)
            house_rate = np.random.normal(self.rh, self.rh * 0.2)
            result_a = self.scenario_a_sell_btc()
            scenario_a_net_values.append(result_a["net_value"])
            result_b = self.scenario_b_borrow_against_btc(btc_rate=btc_rate, house_rate=house_rate, num_paths=1)
            scenario_b_net_values.append(result_b["net_values"][0])
            scenario_b_liquidation.append(result_b["liquidation_occurred_list"][0])
            result_c = self.scenario_c_self_paying_mortgage(btc_rate=btc_rate, house_rate=house_rate, num_paths=1)
            scenario_c_net_values.append(result_c["net_values"][0])
            scenario_c_liquidation.append(result_c["liquidation_occurred_list"][0])

            # Debug output every 100 simulations
            if sim % 100 == 0:
                print(f"Simulation {sim}: A_net={result_a['net_value']:.2f}, B_net={result_b['net_values'][0]:.2f}, C_net={result_c['net_values'][0]:.2f}")

        percentiles = [10, 50, 90]
        return {
            "Scenario A": {
                "Bear Case": np.percentile(scenario_a_net_values, percentiles[0]),
                "Base Case": np.percentile(scenario_a_net_values, percentiles[1]),
                "Bull Case": np.percentile(scenario_a_net_values, percentiles[2]),
                "All Results": scenario_a_net_values,
                "Liquidation Probability": 0
            },
            "Scenario B": {
                "Bear Case": np.percentile(scenario_b_net_values, percentiles[0]),
                "Base Case": np.percentile(scenario_b_net_values, percentiles[1]),
                "Bull Case": np.percentile(scenario_b_net_values, percentiles[2]),
                "All Results": scenario_b_net_values,
                "Liquidation Probability": np.mean(scenario_b_liquidation) * 100
            },
            "Scenario C": {
                "Bear Case": np.percentile(scenario_c_net_values, percentiles[0]),
                "Base Case": np.percentile(scenario_c_net_values, percentiles[1]),
                "Bull Case": np.percentile(scenario_c_net_values, percentiles[2]),
                "All Results": scenario_c_net_values,
                "Liquidation Probability": np.mean(scenario_c_liquidation) * 100
            }
        }

# Helper functions
def run_analysis(inputs):
    """Run the base case and simulations."""
    model = BTCHousingModel(**inputs)
    scenario_a = model.scenario_a_sell_btc()
    scenario_b = model.scenario_b_borrow_against_btc()
    scenario_c = model.scenario_c_self_paying_mortgage()
    simulations = model.simulate_scenarios()
    return model, scenario_a, scenario_b, scenario_c, simulations

def get_recommendation(scenario_a, scenario_b, scenario_c, risk_aversion):
    """Recommend a scenario based on net value and user risk preference."""
    max_net_value = max(scenario_a["net_value"], scenario_b["avg_net_value"], scenario_c["avg_net_value"])
    score_a = scenario_a["net_value"]
    score_b = scenario_b["avg_net_value"] - (risk_aversion / 100) * max_net_value * (scenario_b["liquidation_probability"] / 100)
    score_c = scenario_c["avg_net_value"] - (risk_aversion / 100) * max_net_value * (scenario_c["liquidation_probability"] / 100)
    scores = {"A - Sell BTC to Buy House": score_a, "B - Borrow Against BTC": score_b, "C - Self-Paying Mortgage via Bitcoin Gains": score_c}
    best_scenario = max(scores, key=scores.get)
    return f"Recommended Scenario: {best_scenario} with score {scores[best_scenario]:,.2f}"

def create_distribution_chart(scenario_a, scenario_b, scenario_c):
    """Create a box plot for the distribution of net values."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=scenario_a["net_values"], name="A - Sell BTC to Buy House"))
    fig.add_trace(go.Box(y=scenario_b["net_values"], name="B - Borrow Against BTC"))
    fig.add_trace(go.Box(y=scenario_c["net_values"], name="C - Self-Paying Mortgage via Bitcoin Gains"))
    fig.update_layout(title="Distribution of Net Values Across Simulations", xaxis_title="Scenario", yaxis_title="Net Value ($)")
    return fig

def create_radar_chart(scenario_a, scenario_b, scenario_c):
    """Create a radar chart comparing scenarios."""
    categories = ["Net Value", "Risk Avoidance", "Flexibility", "Leverage"]
    fig = go.Figure()
    max_net_value = max(scenario_a["net_value"], scenario_b["avg_net_value"], scenario_c["avg_net_value"])
    values_a = [
        scenario_a["net_value"] / max_net_value * 100,
        100, 0, 0
    ]
    values_b = [
        scenario_b["avg_net_value"] / max_net_value * 100,
        100 - scenario_b["liquidation_probability"],
        100, 100
    ]
    values_c = [
        scenario_c["avg_net_value"] / max_net_value * 100,
        100 - scenario_c["liquidation_probability"],
        50, 50
    ]
    for values, name in zip([values_a, values_b, values_c], ["A - Sell BTC to Buy House", "B - Borrow Against BTC", "C - Self-Paying Mortgage via Bitcoin Gains"]):
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Scenario Comparison (Radar Chart)")
    return fig

def create_bar_chart(scenario_a, scenario_b, scenario_c):
    """Create a bar chart comparing scenarios."""
    categories = ["Net Value", "Risk Avoidance", "Flexibility", "Leverage"]
    max_net_value = max(scenario_a["net_value"], scenario_b["avg_net_value"], scenario_c["avg_net_value"])
    values_a = [scenario_a["net_value"] / max_net_value * 100, 100, 0, 0]
    values_b = [scenario_b["avg_net_value"] / max_net_value * 100, 100 - scenario_b["liquidation_probability"], 100, 100]
    values_c = [scenario_c["avg_net_value"] / max_net_value * 100, 100 - scenario_c["liquidation_probability"], 50, 50]
    fig = go.Figure()
    for i, category in enumerate(categories):
        fig.add_trace(go.Bar(x=["A - Sell BTC to Buy House", "B - Borrow Against BTC", "C - Self-Paying Mortgage via Bitcoin Gains"], y=[values_a[i], values_b[i], values_c[i]], name=category))
    fig.update_layout(barmode='group', title="Scenario Comparison (Bar Chart)", xaxis_title="Scenario", yaxis_title="Score (0-100)")
    return fig

def create_sensitivity_chart(inputs):
    """Create a sensitivity analysis chart for BTC appreciation rate."""
    btc_rates = np.linspace(-0.2, 1.0, 20)
    net_values_a, net_values_b, net_values_c = [], [], []
    for rate in btc_rates:
        temp_model = BTCHousingModel(**inputs)
        temp_model.rb = rate
        a = temp_model.scenario_a_sell_btc()
        b = temp_model.scenario_b_borrow_against_btc(btc_rate=rate)
        c = temp_model.scenario_c_self_paying_mortgage(btc_rate=rate)
        net_values_a.append(a["net_value"])
        net_values_b.append(b["avg_net_value"])
        net_values_c.append(c["avg_net_value"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_rates*100, y=net_values_a, mode='lines', name='A - Sell BTC to Buy House'))
    fig.add_trace(go.Scatter(x=btc_rates*100, y=net_values_b, mode='lines', name='B - Borrow Against BTC'))
    fig.add_trace(go.Scatter(x=btc_rates*100, y=net_values_c, mode='lines', name='C - Self-Paying Mortgage via Bitcoin Gains'))
    fig.update_layout(title='Sensitivity of Base Net Value to BTC Appreciation Rate', xaxis_title='BTC Annual Appreciation Rate (%)', yaxis_title='Base Net Value ($)')
    return fig

# Streamlit UI
st.title("Bitcoin Housing Strategy Analyzer")
st.write("Compare strategies for using Bitcoin to purchase a house.")

with st.expander("Bitcoin Parameters", expanded=True):
    initial_btc = st.number_input("Initial BTC Holdings", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    initial_btc_price = st.number_input("Initial BTC Price ($)", min_value=1000, max_value=1000000, value=60000)
    btc_basis_price = st.number_input("BTC Basis Price ($)", min_value=0, max_value=1000000, value=30000)
    btc_appreciation_rate = st.slider("BTC Annual Appreciation Rate (%)", -20, 100, 10, step=1) / 100
    btc_volatility = st.slider("BTC Annual Volatility (%)", 10, 100, 50, step=1) / 100

with st.expander("House Parameters"):
    initial_house_price = st.number_input("Initial House Price ($)", min_value=50000, max_value=10000000, value=500000, step=10000)
    house_appreciation_rate = st.slider("House Annual Appreciation Rate (%)", 0, 20, 3, step=1) / 100
    house_buying_fee = st.slider("House Buying Fee (%)", 0, 10, 3, step=1) / 100
    house_holding_cost = st.slider("Annual Holding Cost (%)", 0, 5, 1, step=1) / 100

with st.expander("Financial Parameters"):
    inflation_rate = st.slider("Inflation Rate (%)", 0, 10, 2, step=1) / 100
    tax_rate = st.slider("Capital Gains Tax Rate (%)", 0, 50, 20, step=1) / 100
    loan_interest_rate = st.slider("Loan Interest Rate (%)", 0, 20, 5, step=1) / 100
    ltv_ratio = st.slider("Loan-to-Value Ratio (%)", 10, 90, 50, step=5) / 100
    risk_aversion = st.slider("Risk Aversion (0 = Neutral, 100 = Averse)", 0, 100, 50, step=1)

with st.expander("Simulation Parameters"):
    time_horizon = st.number_input("Time Horizon (Years)", min_value=1, max_value=30, value=10, step=1)
    btc_selling_fee = st.slider("BTC Selling Fee (%)", 0, 5, 1, step=1) / 100

if any(value <= 0 for value in [initial_btc, initial_btc_price, btc_basis_price, initial_house_price]):
    st.error("All prices and holdings must be positive.")
    st.stop()

inputs = {
    "initial_btc": initial_btc, "initial_btc_price": initial_btc_price, "btc_basis_price": btc_basis_price,
    "initial_house_price": initial_house_price, "btc_appreciation_rate": btc_appreciation_rate,
    "house_appreciation_rate": house_appreciation_rate, "inflation_rate": inflation_rate,
    "btc_volatility": btc_volatility, "btc_selling_fee": btc_selling_fee, "house_buying_fee": house_buying_fee,
    "house_holding_cost": house_holding_cost, "tax_rate": tax_rate, "loan_interest_rate": loan_interest_rate,
    "ltv_ratio": ltv_ratio, "time_horizon": time_horizon
}
model, scenario_a, scenario_b, scenario_c, simulations = run_analysis(inputs)

tab1, tab2, tab3 = st.tabs(["Scenario Details", "Comparison", "Charts"])

with tab1:
    st.header("Scenario Details")
    for scenario in [scenario_a, scenario_b, scenario_c]:
        st.subheader(scenario["scenario"])
        if scenario["scenario"] == "A - Sell BTC to Buy House":
            st.write(f"Can Afford House: {'Yes' if scenario['can_afford_house'] else 'No'}")
            st.write(f"Net Proceeds After Tax: ${scenario['net_proceeds_after_tax']:,.2f}")
            st.write(f"House Cost: ${scenario['house_cost']:,.2f}")
            st.write(f"Remaining Cash: ${scenario['remaining_cash']:,.2f}")
            st.write(f"Total Holding Costs (Undiscounted): ${scenario['total_holding_costs']:,.2f}")
            st.write(f"Holding Costs (Present Value): ${scenario['holding_costs_pv']:,.2f}")
            st.write(f"Final House Value: ${scenario['final_house_value']:,.2f}")
            st.write(f"Net Value: ${scenario['net_value']:,.2f}")
        else:
            st.write(f"Can Afford House: {'Yes' if scenario.get('can_afford_house', True) else 'No'}")
            st.write(f"Initial Loan/Amount: ${scenario.get('loan_amount', scenario.get('initial_loan', 0)):,.2f}")
            st.write(f"House Cost: ${scenario.get('house_cost', 0):,.2f}")
            st.write(f"Remaining Cash: ${scenario.get('remaining_cash', 0):,.2f}")
            st.write(f"Average Net Value: ${scenario['avg_net_value']:,.2f}")
            st.write(f"Liquidation Probability: {scenario['liquidation_probability']:.2f}%")

with tab2:
    st.header("Comparison")
    df = pd.DataFrame({
        "Scenario": ["A - Sell BTC to Buy House", "B - Borrow Against BTC", "C - Self-Paying Mortgage via Bitcoin Gains"],
        "Base Net Value ($)": [scenario_a["net_value"], scenario_b["avg_net_value"], scenario_c["avg_net_value"]],
        "Liquidation Risk (%)": [0, scenario_b["liquidation_probability"], scenario_c["liquidation_probability"]],
        "Bear Case ($)": [simulations["Scenario A"]["Bear Case"], simulations["Scenario B"]["Bear Case"], simulations["Scenario C"]["Bear Case"]],
        "Bull Case ($)": [simulations["Scenario A"]["Bull Case"], simulations["Scenario B"]["Bull Case"], simulations["Scenario C"]["Bull Case"]]
    })
    st.table(df.style.format({"Base Net Value ($)": "{:,.2f}", "Bear Case ($)": "{:,.2f}", "Bull Case ($)": "{:,.2f}", "Liquidation Risk (%)": "{:.2f}"}))

    st.subheader("Recommendation")
    recommendation = get_recommendation(scenario_a, scenario_b, scenario_c, risk_aversion)
    st.write(recommendation)

with tab3:
    st.header("Charts")
    fig_dist = create_distribution_chart(scenario_a, scenario_b, scenario_c)
    st.plotly_chart(fig_dist)
    
    fig_radar = create_radar_chart(scenario_a, scenario_b, scenario_c)
    st.plotly_chart(fig_radar)
    
    fig_bar = create_bar_chart(scenario_a, scenario_b, scenario_c)
    st.plotly_chart(fig_bar)
    
    with st.expander("Sensitivity Analysis"):
        st.subheader("Sensitivity to BTC Appreciation Rate")
        fig_sens = create_sensitivity_chart(inputs)
        st.plotly_chart(fig_sens)
