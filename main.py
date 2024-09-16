import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import beta
import ruptures as rpt
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.express as px
import plotly.graph_objects as go


seed = st.slider("Random Seed (set for reproducibility)", 1, 1000, 24)
# Function to generate a binomial time series with changing trend or sudden shock
def generate_time_series(N, p_initial, date, change_type, change_date=None,change=None):
    np.random.seed(seed)  # For reproducibility
    p = p_initial
    series = []    
    index = 0
    for day in date:

        if change_date is not None:
            if day.date() >= change_date:
                if change_type == "Trend Change":
                    p += change  # Slowly increase probability in the second half
                elif (change_type == "Sudden Shock") and (day.date() == change_date):
                    p = p_initial+change  # Sudden jump in probability at shock day
        
        p = max(min(p, 1), 0)  # Ensure probability remains between 0 and 1
        series.append(pd.DataFrame({'Success':np.random.binomial(N, p),
                                    'Responses':N,
                                    'True P':p,
                                    'Date':pd.to_datetime(day)},index=[index]))
        index += 1
    
    return pd.concat(series).sort_values('Date')

# Function to calculate confidence interval
def calculate_confidence_interval(proportion, n, confidence=0.95):
    std_err = np.sqrt((proportion * (1 - proportion)) / n)
    margin_of_error = std_err * norm.ppf((1 + confidence) / 2)
    return proportion - margin_of_error, proportion + margin_of_error

# Function to aggregate time series data by frequency
def aggregate_data(time_series, freq='D'):
    if (freq == 'W') | (freq == 'M'):
        counts = time_series[['Date','Responses','Success']].resample(freq, on='Date').sum()
        p_values = time_series[['Date','True P']].resample(freq, on='Date').mean()
        return counts.merge(p_values,left_index=True,right_index=True)

    return time_series.set_index('Date')

# Z-test for proportions (one-tailed test: H1: p1 > p2)
def z_test_for_proportions_one_tailed(success1, n1, success2, n2):
    p1 = success1 / n1
    p2 = success2 / n2
    pooled_p = (success1 + success2) / (n1 + n2)
    std_err = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    z = (p1 - p2) / std_err
    p_value = 1 - norm.cdf(z)  # One-tailed test for p1 > p2
    return z, p_value
    
# Z-test for proportions
def z_test_for_proportions(success1, n1, success2, n2):
    p1 = success1 / n1
    p2 = success2 / n2
    pooled_p = (success1 + success2) / (n1 + n2)
    std_err = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    z = (p1 - p2) / std_err
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    return z, p_value

# Function to apply Bayesian estimation (Beta-Binomial)
def bayesian_estimation(successes, responses, alpha_prior=1, beta_prior=1):
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + responses - successes
    return alpha_post, beta_post

def calculate_power(p1, p2, n1, n2, alpha=0.05):
    z_alpha = norm.ppf(1 - alpha / 2)  # Critical value for two-tailed test
    std_err_diff = np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    effect_size = abs(p1 - p2) / std_err_diff
    power = 1 - norm.cdf(z_alpha - effect_size)
    return power

def apply_ewm(data, period):
    successes = data.Success.ewm(span=period).sum()
    responses = data.Responses.ewm(span=period).sum()
    data['EWMResponses'] = responses
    data['EWMSmoothed'] = successes/responses
    return data

# CUSUM change detection using ruptures
def apply_cusum(time_series):
    algo = rpt.Binseg(model="l2").fit(time_series['Proportion'].values)
    penalty =0.01
    return algo.predict(pen=penalty)

# App UI
st.title("Random Binomial Time Series Generator")

# User inputs
days = st.slider("Number of days", 10, 1000, 90)
N = st.number_input("Daily N", min_value=1, value=30)
p_initial = st.slider("Initial probability (p)", 0.01, 1.0, 0.5)
change_type = st.selectbox("Select the type of change", ["No Change", "Trend Change", "Sudden Shock"])

#create the date range
date = pd.date_range(start='2021-01-01', periods=days)

shock_day = int(days/2)
change= None
change_date = None
if change_type == "Sudden Shock":
    change_date = st.date_input("Date of sudden shock", value=date[len(date)//2], min_value=min(date),
                                     max_value=date[len(date)//2])
    change = st.slider("Shock size(p)", 0.01, 1.0, 0.05)
if change_type == "Trend Change":
    change_date = st.date_input("Date of trend change", value=date[len(date)//2], min_value=min(date),
                                     max_value=date[len(date)//2])
    change = st.slider("Trend Change Increment (per day)", 0.0001, 0.01, 0.001, step=0.0001,format="%f")
    

# Generate the time series
time_series = generate_time_series(N, p_initial, date, change_type, change_date,change)

freq = st.selectbox("Select data frequency", ["Daily", "Weekly", "Monthly"])

# Aggregate time series data based on frequency
if freq == "Weekly":
    df = aggregate_data(time_series, 'W')
    f = 'Weeks'
elif freq == "Monthly":
    df = aggregate_data(time_series, 'M')
    f = 'Months'
else:
    df = aggregate_data(time_series, 'D')
    f = 'Days'

# Calculate the proportion of successes
df['Proportion'] = df['Success'] / df['Responses']


# Select the smoothing method
smoothing_method = st.selectbox("Select smoothing method", ["EWM","Bayesian"])

# Initialize Kalman and CUSUM variables
kalman_smoothed = None
cusum_detected_changes = []

days = len(df)
# Apply the selected method
if smoothing_method == "Bayesian":
    with st.expander("Click here for a Bayesian"):
        st.write("""Bayesian statistics allow for the inclusion of prior knowledge, in this instance we are using X 
        days worth of previous data as our prior. This is under the assumption that most movement is sample error.
        Including a prior will drag any daily values back to the recent mean. If we include too many days this will override
        any sort of variance.""")
    # Configurable window size for updating the prior
    window_size = st.slider(f"Select window size for prior updates ({f})", 1, int(days/2), int(days/4))

    # Initialize the posterior columns
    df['Alpha_Post'] = 0
    df['Beta_Post'] = 0
    
    # Set initial prior values
    alpha_prior = 0#st.number_input("Initial Alpha prior (a)", min_value=0.1, value=1.0)
    beta_prior = 0#st.number_input("Initial Beta prior (b)", min_value=0.1, value=1.0)
    
    # Perform Bayesian estimation with dynamic prior updating
    for i in range(len(df)):
        # Determine the window of the last N days for updating the prior
        start_idx = max(0, i - window_size)
        prior_window = df.iloc[start_idx:i]
    
        # Sum successes and responses from the last N days to update the prior
        successes_last_n = prior_window['Success'].sum()
        responses_last_n = prior_window['Responses'].sum()
    
        # Update the prior based on the last N days' data
        alpha_dynamic_prior = (alpha_prior*(max(0, window_size-1)/window_size)) + successes_last_n
        beta_dynamic_prior = (beta_prior*(max(0, window_size-1)/window_size)) + responses_last_n - successes_last_n
    
        # Compute the posterior for the current time period
        df.at[df.index[i], 'Alpha_Post'], df.at[df.index[i], 'Beta_Post'] = bayesian_estimation(
            df.iloc[i]['Success'], df.iloc[i]['Responses'], alpha_dynamic_prior, beta_dynamic_prior)
    
    # Calculate the mean and 95% credible interval for each period's posterior distribution
    df['Posterior_Mean'] = df['Alpha_Post'] / (df['Alpha_Post'] + df['Beta_Post'])
    df['Lower_CI'] = beta.ppf(0.025, df['Alpha_Post'], df['Beta_Post'])
    df['Upper_CI'] = beta.ppf(0.975, df['Alpha_Post'], df['Beta_Post'])

# Apply the selected method
elif smoothing_method == "EWM":
    with st.expander("Click here for a EWM explainer"):
        st.write("""Exponentially Weighted Moving Average (EWMA) smoothing is a technique used 
        to smooth time series data by giving more weight to recent observations while gradually
        decreasing the weight for older observations. The key idea is that more recent data points
        are more relevant for forecasting or trend analysis than older ones.""")
    period = st.slider(f"Weighting span ({f})", 1, int(days/2), 1)
    df = apply_ewm(df,period)
    df['Lower_CI'],df['Upper_CI'] = calculate_confidence_interval(df.EWMSmoothed,df.EWMResponses)
#elif smoothing_method == "CUSUM":
#    cusum_detected_changes = apply_cusum(df)

show_ci = st.checkbox('Show CI')
# Plot the data
fig = go.Figure()

# Plot the observed proportions
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['Success'] / df['Responses'], 
    mode='markers', 
    name='Observed Proportion',
    marker=dict(color='gray', size=6, opacity=0.6)
))

fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['True P'], 
        mode='lines', 
        name='True Proportion',
        line=dict(color='black')
    ))

if smoothing_method == "Bayesian":
    # Plot Bayesian results
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Posterior_Mean'], 
        mode='lines', 
        name='Posterior Mean (Bayesian)',
        line=dict(color='blue')
    ))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1], 
            y=df['Upper_CI'].tolist() + df['Lower_CI'].tolist()[::-1], 
            fill='toself', 
            fillcolor='lightblue', 
            line=dict(color='lightblue'),
            name='95% Credible Interval',
            opacity=0.5
        ))
elif smoothing_method == "EWM":
    # Plot Kalman filter results
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['EWMSmoothed'], 
        mode='lines', 
        name='Exponentially Weighted MA',
        line=dict(color='blue')
    ))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=df.index.tolist() + df.index.tolist()[::-1], 
            y=df['Upper_CI'].tolist() + df['Lower_CI'].tolist()[::-1], 
            fill='toself', 
            fillcolor='lightblue', 
            line=dict(color='lightblue'),
            name='95% Confidence Interval',
            opacity=0.5
        ))

elif smoothing_method == "CUSUM":
    st.write(cusum_detected_changes)
    # Highlight CUSUM detected changes
    for change in cusum_detected_changes:
        if change<len(df):
            fig.add_vline(x=df.index[change-1], line=dict(color='red', dash='dash'), name='CUSUM Change Point',legend=True)

if change_type!='No Change':
    fig.add_vline(x=change_date, line=dict(color='black', dash='dash'), name='True Change Point')
    #This invisible line will addthe vline to the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                         line=dict(color="black", dash="dash"), 
                         showlegend=True, name="True Change Point"))
# Update plot layout
fig.update_layout(
    title=f"{smoothing_method} Method on Binomial Time Series",
    xaxis_title="Date",
    yaxis_title="Probability (p)",
    legend_title="Legend"
)

st.plotly_chart(fig)

# Select periods for comparison

st.write("Select two periods to compare by date:")

# Allow user to select start and end dates for both periods
period1_start_date = st.date_input("Start date of Period 1", value=df.index.min(), min_value=df.index.min(), max_value=df.index.max())
period1_end_date = st.date_input("End date of Period 1", value=df.index[len(df)//2], min_value=period1_start_date, max_value=df.index.max())

period2_start_date = st.date_input("Start date of Period 2", value=df.index[len(df)//2], min_value=df.index.min(), max_value=df.index.max())
period2_end_date = st.date_input("End date of Period 2", value=df.index.max(), min_value=period2_start_date, max_value=df.index.max())

# Filter data based on selected dates
period1_data = df.loc[period1_start_date:period1_end_date]
period2_data = df.loc[period2_start_date:period2_end_date]


true_diff = period2_data['True P'].mean() - period1_data['True P'].mean()

# Calculate total successes and responses for each period
success1 = period1_data['Success'].sum()
n1 = period1_data['Responses'].sum()

success2 = period2_data['Success'].sum()
n2 = period2_data['Responses'].sum()

# Calculate proportions for each period
p1 = success1 / n1
p2 = success2 / n2

# Perform Z-test for proportions
z_stat, p_value = z_test_for_proportions_one_tailed(success2, n2, success1, n1)

# Calculate confidence intervals for each proportion
ci_low1, ci_high1 = calculate_confidence_interval(p1, n1)
ci_low2, ci_high2 = calculate_confidence_interval(p2, n2)

# Display results
st.write(f"Proportion in Period 1: {p1:.4f} (CI: [{ci_low1:.4f}, {ci_high1:.4f}])")
st.write(f"Proportion in Period 2: {p2:.4f} (CI: [{ci_low2:.4f}, {ci_high2:.4f}])")
st.write(f"Z-statistic: {z_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")

desired_confidence = st.slider("Desired Confidence Level", 0.80, 0.99, 0.95)
alpha = 1 - desired_confidence

if p_value < alpha:
    st.write(f"The difference between the two periods is statistically significant at confidence level {desired_confidence}. At this confidence level we would expect type II errors 5% of the time")
else:
    st.write(f"The difference between the two periods is not statistically significant at confidence level {desired_confidence}.")


# Power Calculation

power = calculate_power(period1_data['True P'].mean(), period2_data['True P'].mean(), n1, n2, alpha)

# Display power calculation result
if change_type != 'No Change':
    st.write(f"Statistical Power of the test: {power:.4f}")
    with st.expander("Statistical Power"):
        st.write(f"""With a true difference of {(true_diff):.3f} and sample sizes of {n1} and {n2} we would spot a significant
    difference {(power*100):.2f}% of the time, leading to Type I errors {((1-power)*100):.2f}% of the time""")

# Explanation of Statistical Concepts with Expandable Text
with st.expander("Click here for a detailed explanation of Sampling Error, Hypothesis Testing, Type I and II Errors, Power, and Confidence"):
    st.write("""
    ### Sampling Error:
    When you take a sample from a larger group, the results you observe might not exactly represent the true values 
    in the population. This is called **sampling error**—the natural variation that occurs when you look at a 
    subset of the whole population.
    
    - **Example**: Across a week a store greets 85% of their customers, by asking customers a single varying question we get a subset 
    of these customers. Using simulation we can repeat this sampling 1000's of times allowing us to see the variation we might expect.""")

    st.plotly_chart(px.histogram(np.random.binomial(n=100,p=0.85,size=10000),
                                                    title="Sample error for p=85% and sample size=100"))
    
    st.write("""Here we can see that even though the true value of greetings is always 85% the range of values we might
    see from our sample go from \<75% to \>95%.
    
Because of sampling error, small differences between two groups might not be meaningful, which is why statistical 
tests are needed to determine whether the differences you observe reflect real underlying differences or are 
just due to chance.

### Hypothesis Testing:
    **Hypothesis testing** is a statistical method used to decide whether the difference we observe between two groups 
    is significant or just due to random chance. In hypothesis testing, we typically define two competing hypotheses:

    - **Null Hypothesis (H₀)**: Assumes that there is no real difference between the groups (e.g., both stores greet the same percentage of customers).
    - **Alternative Hypothesis (H₁)**: Assumes that there is a real difference between the groups (e.g., one store greets more customers than the other).

    We use sample data to decide whether we can reject the null hypothesis in favor of the alternative hypothesis.

    - **Example**: You want to determine whether one store greets a higher proportion of customers than another. You collect a sample and 
      perform a statistical test. Based on the test, you either reject the null hypothesis (conclude there is a significant difference) 
      or fail to reject it (conclude there is no strong evidence of a difference).

### Type I Error (False Positive):
A **Type I error** occurs when you conclude that there is a difference in the proportion of customers helped 
when, in reality, no real difference exists. In this case, a Type I error would happen if you incorrectly 
determine that one store is helping a higher proportion of customers than another, 
when both groups are actually performing the same.

- **Example**: 2 stores are greeting customers 85% of the time, but the scores seen over a week are 80 for one 
store and 90 for the other. Based on this difference, you conclude that one 
  store is more effective, even though the difference is just due to random chance (sampling error).

### Type II Error (False Negative):
A **Type II error** happens when you fail to detect a real difference that exists. 
For the customer help example, a Type II error occurs if you conclude that the proportion of customers helped 
by both teams is the same, even though one team is truly helping a larger proportion of customers.

- **Example**: One store improves and greets 90% of customers over a week while the other store maintains at 85%, 
but the sample proportions both show 87%.You conclude that there is no difference between the teams, 
even though, with more data, you might have found that one team consistently greets more customers.

### Power:
**Power** is the probability of correctly detecting a real difference in the proportion of customers helped 
when it exists. In this context, high power means that you are more likely to detect this difference.

- **Example**: As you collect more data the sampling error shinks, while we might expect a value to be &plusmn;10%
around the true proportion when we have 100 responses, we would expect this to shink to 2-3% with 1000 responses.

### Confidence:
**Confidence** refers to how certain we are about the results of our test. In hypothesis testing, a confidence 
interval gives a range of values within which we believe the true proportion of customers helped lies. If we use 
a 95% confidence level, it means that if we repeated the experiment many times, 95% of the time, the true 
difference in proportions would fall within the interval we calculated.
""")
