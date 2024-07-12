import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import cloudpickle
import pytensor.tensor as tt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
import os
import re
import pytensor.tensor as pt
import seaborn as sns
from scipy.optimize import curve_fit

# Transformation Functions
def clean_names(names_list):
    cleaned_names = []
    for name in names_list:
        name = name.replace('%', 'percent').replace('ö', 'o')
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        name = re.sub(r'\W+', '_', name.lower())
        name = re.sub(r'__+', '_', name)
        cleaned_names.append(re.sub(r'_$', '', name))
    return cleaned_names

def process_excel(file_path, tariff, condition_str='dv'):
    """
    Reads an Excel file, identifies columns relevant based on a 'dv' marker in a specified row,
    returns a DataFrame with these columns and 'date' set as the index, dropping any duplicates.
    It also prints a message if there are duplicated rows.

    Parameters:
    - file_path: str, the path to the Excel file.
    - tariff: str, a string that helps identify the relevant row.
    - condition_str: searches for this string as condition to use row in dataset.

    Returns:
    - pd.DataFrame: DataFrame with the relevant columns, with 'date' as the index.
    """
    # Initial read to find the important row
    df = pd.read_excel(file_path)  # This assumes the Excel file has a detectable header
    
    # Identify the important row based on the 'notes' column and the tariff
    row_important = df[df['notes'] == f'relevance'].index.item()
    
    # Determine which columns are marked as 'dv'
    condition = df.iloc[row_important] == condition_str
    
    # Filter column names based on the 'dv' condition, excluding NaNs
    names = [x for x in df.iloc[8][condition].tolist() if not pd.isna(x)]
    
    # Re-read the Excel file with the correct header row
    dat = pd.read_excel(file_path, header=9)
    
    # Select the relevant columns plus 'date'
    dat = dat[names + ['date']]
    
    # Check for duplicated column names and drop them
    dat = dat.loc[:,~dat.columns.duplicated()]
    
    # Set 'date' as the index
    dat = dat.set_index('date')
    
    return dat


def remove_duplicates_preserve_order(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def scale_data(data: pd.DataFrame, type="MinMax"):
    scaler = MinMaxScaler() if type == "MinMax" else MaxAbsScaler()
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def inverse_transform_data(scaler, data):
    return scaler.inverse_transform(data)

def scale_theano(x):
    return x / tt.max(x)

# Plotting Functions
def plot_correlation_heatmap(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, fmt=".2f", annot=True, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, annot_kws={"size": 7}, ax=ax)
    sns.heatmap(corr.fillna(0), mask=mask | ~(np.isnan(corr)), cmap="coolwarm", square=True, linewidths=0.5, cbar=False, annot=False, ax=ax)
    ax.tick_params(axis="y", rotation=0)
    plt.show()
    plt.rcdefaults()

def plot_scatter_plotly(x, y):
    return px.scatter(x=x, y=y)

def plot_data(model_df, date_column='date', scale_data=False, plot_height=800, columns_to_plot=None):
    """
    Plots specified columns of a dataframe over time, with an option to scale the data.

    Parameters:
    - model_df: Pandas DataFrame with a date column and other numeric columns to plot.
    - date_column: The name of the column containing date information.
    - scale_data: Boolean flag indicating whether to scale the data (True) or not (False).
    - plot_height: The height of the plot in pixels.
    - columns_to_plot: List of column names to be plotted. If None, all columns except the date column are plotted.
    """
    # Convert the date column to datetime format if it's not already
    model_df[date_column] = pd.to_datetime(model_df[date_column])

    # If no specific columns are specified, plot all columns except the date column
    if columns_to_plot is None:
        columns_to_plot = [col for col in model_df.columns if col != date_column]

    # Optionally scale the data
    if scale_data:
        scaler = StandardScaler()
        # Scale only the columns that are to be plotted and are not the date column
        columns_to_scale = [col for col in columns_to_plot if col in model_df.columns and col != date_column]
        scaled_df = model_df.copy()  # Create a copy to keep the original dataframe unchanged
        scaled_df[columns_to_scale] = scaler.fit_transform(model_df[columns_to_scale])
    else:
        scaled_df = model_df

    # Create a plotly figure
    fig = go.Figure()

    # Iterate over the specified columns to add them to the plot
    for column in columns_to_plot:
        if column in scaled_df.columns:  # Check if the column exists in the dataframe
            fig.add_trace(go.Scatter(x=scaled_df[date_column], y=scaled_df[column], mode='lines', name=column))

    # Update the layout with a dynamic height
    fig.update_layout(title='Plot of Selected Columns Over Time',
                      xaxis_title='Date',
                      yaxis_title='Values' if not scale_data else 'Scaled Values',
                      legend_title='Columns',
                      height=plot_height)

    # Show the figure
    fig.show()

def plot_predictions(actuals, predictions, title):
    plt.plot(actuals, label="Actuals")
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.title(title)

def plotly_line(df):
    return px.line(x=df.index, y=df.values)

# Efficiency and Contribution Functions
def compute_mean_control(trace, var):
    var = var.rstrip("_")
    return trace.posterior[f"contribution_{var}"].mean(axis=1).mean(axis=0).to_numpy()


def compute_contributions(trace, media_vars, controls_positive, controls_negative, df, predictions, intercept=True, trend=True, seasonality=True):

    unadj_contributions = pd.DataFrame(
        index=df.index
    )
    # Intercept contributions
    if intercept:
        unadj_contributions["Intercept"] = trace["posterior"]["Intercept"].mean(axis=1).mean(axis=0).to_numpy()
    # Trend contributions
    if trend:
        unadj_contributions["trend"] = trace["posterior"]["trend_contribution"].mean(axis=1).mean(axis=0).to_numpy()
    # Seasonality contributions
    if seasonality:
        if "fourier_contribution_weekly" in trace["posterior"]:
            unadj_contributions["fourier_contribution_weekly"] =trace["posterior"]["fourier_contribution_weekly"].mean(axis=1).mean(axis=0).to_numpy()
        if "fourier_contribution_monthly" in trace["posterior"]:
            unadj_contributions["fourier_contribution_monthly"] =trace["posterior"]["fourier_contribution_monthly"].mean(axis=1).mean(axis=0).to_numpy()
        if "fourier_contribution_annual" in trace["posterior"]:
            unadj_contributions["fourier_contribution_annual"] =trace["posterior"]["fourier_contribution_annual"].mean(axis=1).mean(axis=0).to_numpy()

    # Media channel contributions
    for channel in media_vars:
        unadj_contributions[channel] = trace["posterior"]["channel_contributions"].sel(channel=channel).mean(axis=1).mean(axis=0).to_numpy()

    # Control contributions
    for i in controls_positive:
        unadj_contributions[i] = trace["posterior"]["control_contributions_positive"].sel(control_positive=i).mean(axis=1).mean(axis=0).to_numpy()
    # Control contributions
    for i in controls_negative:
        unadj_contributions[i] = trace["posterior"]["control_contributions_negative"].sel(control_negative=i).mean(axis=1).mean(axis=0).to_numpy()

    # Adjusting contributions based on the target variable
    adj_contributions = unadj_contributions.div(
        unadj_contributions.sum(axis=1), axis=0
    ).mul(predictions, axis=0)

    return unadj_contributions, adj_contributions


def get_efficiencies(coefs):
    coefficients = coefs.loc[[i for i in coefs.index if "beta_channel" in i or "beta_control_positive" in i or "beta_control_negative"in i]]
    coefficients["Normalized"] = coefficients["mean"].abs() / coefficients["mean"].abs().sum() * 100 * coefficients["mean"].apply(lambda x: 1 if x >= 0 else -1)
    coefficients = coefficients.round(2).sort_values("Normalized", ascending=True)
    colors = ['green' if "Grip" in var_name else 'blue' for var_name in coefficients.index]
    fig = px.bar(coefficients, x="Normalized", y=coefficients.index, orientation="h", text="Normalized", color=colors)
    fig.update_layout(xaxis_title="Efficiency in Driving Brand", yaxis_title="Features", title="Normalized Coefficients")
    fig.update_traces(textangle=0, texttemplate='%{x:.2f}')
    return coefficients, fig

def get_efficiencies_corrected(coefs, neg_vars, intercept=False):
    if intercept:
        coefficients = coefs.loc[[i for i in coefs.index if "beta_channel" in i or "beta_control_positive" in i or "beta_control_negative" in i or "Intercept" in i or "beta_trend" in i]]
    else:
        coefficients = coefs.loc[[i for i in coefs.index if "beta_channel" in i or "beta_control_positive" in i or "beta_control_negative"in i or "beta_trend" in i]]
    for var in neg_vars:
        coefficients.loc[coefficients.index.str.contains(var), "mean"] *= -1

    fourier_coefficients = coefs.loc[coefs.index.str.contains("beta_fourier")]

    mean_fourier = fourier_coefficients["mean"].mean()
    coefficients.loc["beta_fourier"] = mean_fourier
    
    coefficients.index = coefficients.index.map(lambda x: x[x.find('[')+1:x.find(']')] if '[' in x and ']' in x else x)

    coefficients["Normalized"] = coefficients["mean"].abs() / coefficients["mean"].abs().sum() * 100 * coefficients["mean"].apply(lambda x: 1 if x >= 0 else -1)
    coefficients = coefficients.round(2).sort_values("Normalized", ascending=True)
    colors = ['green' if "Grip" in var_name else 'blue' for var_name in coefficients.index]
    fig = px.bar(coefficients, x="Normalized", y=coefficients.index, orientation="h", text="Normalized", color=colors)
    fig.update_layout(xaxis_title="Efficiency in Driving Brand", yaxis_title="Features", title="Normalized Coefficients", height=600)  # Adjust height here
    fig.update_traces(textangle=0, texttemplate='%{x:.2f}')
    return coefficients, fig


def plot_contributions(df, keep_intercept=True):
    if not keep_intercept:
        if 'Intercept' in df.columns:
            df = df.drop(columns=['Intercept', 'trend'])
    mean_contributions = pd.DataFrame(df.mean())
    mean_contributions.columns = ["mean"]
    contributions = mean_contributions.abs().mean(axis=1) / mean_contributions.abs().mean(axis=1).sum() * 100 * mean_contributions["mean"].apply(lambda x: 1 if x >= 0 else -1)
    
    # Ensure contributions is a DataFrame (important for returning)
    contributions_df = pd.DataFrame(contributions, columns=["Contribution"])
    contributions_df['Features'] = contributions_df.index
    
    # Plotting using Plotly
    fig = px.bar(contributions_df.sort_values(by="Contribution"), 
                 x='Contribution', y='Features', orientation='h',
                 labels={'Contribution': 'Percentage Contribution', 'Features': 'Features'},
                 title='Contribution of Features')
    fig.update_layout(height=600)  # Set the height of the figure
    fig.update_traces(textangle=0, texttemplate='%{x:.2f}')

    # Return both the DataFrame and the figure
    return contributions_df, fig

    
    contributions = pd.DataFrame(contributions, columns=["contribution"])
    contributions = contributions.sort_values("contribution", ascending=True)
    fig = px.bar(contributions, x="contribution", y=contributions.index, orientation="h", text="contribution")
    fig.update_layout(xaxis_title="Contribution To Sales", yaxis_title="Features", height=800)
    fig.update_traces(textangle=0, texttemplate="%{x:.2f}")
    return contributions, fig





def plot_correlation_matrix(df, vars):
    filtered_df = df[vars]
    corr_matrix = filtered_df.corr(method="spearman")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(2*len(vars), 2*len(vars)), facecolor='white')  # Ensure background color is white

    # Define custom colormap: red for negative, green for positive, white for neutral
    cmap = sns.diverging_palette(10, 133, sep=80, n=7)

    # Adjust font sizes based on the number of variables
    if len(vars) > 10:
        annot_size = 25
        axis_fontsize = 25
    else:
        annot_size = 15
        axis_fontsize = 15

    # Draw the heatmap with the mask and custom settings
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f",
                annot_kws={"size": annot_size, "color": "black"},
                xticklabels=vars, yticklabels=vars)  # Pass vars for explicit labels

    # Set the background of the masked area to white
    plt.gca().set_facecolor('white')

    # Rotate the x-axis labels for better readability and adjust font size
    plt.xticks(rotation=75, fontsize=axis_fontsize)  
    plt.yticks(rotation=0, fontsize=axis_fontsize)

    plt.title("Correlation Matrix", fontsize=12)  # Adjust title size if needed
    plt.show()
    return plt




def geometric_adstock(x, alpha, l_max, normalize):
    """Vectorized geometric adstock transformation."""
    cycles = [
        pt.concatenate(tensor_list=[pt.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]])
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    x_cycle = pt.transpose(x=x_cycle, axes=[1, 2, 0])
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    w = pt.transpose(w)[None, ...]
    w = w / pt.sum(w, axis=2, keepdims=True) if normalize else w
    return pt.sum(pt.mul(x_cycle, w), axis=2)


def logistic_saturation(x, lam):
    """Logistic saturation transformation."""
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))

def hill_transform_theano(x, alpha, gamma):
    return tt.pow(x, alpha) / (tt.pow(x, alpha) + tt.pow(gamma, alpha))

def saturate(x, a):
    return (1 - tt.exp(-a * x)) / (1 + tt.exp(-a * x))

def delayed_adstock(alpha, theta, L):

    return alpha**((np.ones(L).cumsum()-1)-theta)**2

def carryover(x, alpha, L, theta = None, func='delayed'):
    transformed_x = []
    if func=='geo':
        weights = geoDecay(alpha, L)
        
    elif func=='delayed':
        weights = delayed_adstock(alpha, theta, L)
    
    for t in range(x.shape[0]):
        upper_window = t+1
        lower_window = max(0,upper_window-L)
        current_window_x = x[:upper_window]
        t_in_window = len(current_window_x)
        if t < L:
            new_x = (current_window_x*np.flip(weights[:t_in_window], axis=0)).sum()
            transformed_x.append(new_x/weights[:t_in_window].sum())
        elif t >= L:
            current_window_x = x[upper_window-L:upper_window]
            ext_weights = np.flip(weights, axis=0) 
            new_x = (current_window_x*ext_weights).sum()
            transformed_x.append(new_x/ext_weights.sum())
            
    return np.array(transformed_x)

# Model Serialization and Deserialization
def read_model_from_pickle(path, name):
    model_file = open(os.path.join(path, name + ".pkl"), "rb")
    model = cloudpickle.load(model_file)
    return model

def dump_model_to_pickle(model, trace, media_vars, control_vars, scaler, cols_to_scale, path, name):
    model_data = cloudpickle.dumps({
        "model": model,
        "trace": trace,
        "media_vars": media_vars,
        "control_vars": control_vars,
        "scaler": scaler,
        "cols_to_scale": cols_to_scale
    })
    with open(os.path.join(path, name + ".pkl"), "wb") as file:
        file.write(model_data)

def plot_saturation_curve(contributions_series, media_series, channel_name):

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Normalize the media spend data to improve fitting
    x_data = media_series.values
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    x_data_normalized = (x_data - x_min) / (x_max - x_min)
    y_data = contributions_series.values

    # Initial guesses for L, k, x0
    initial_guess = [max(y_data), 1, 0.5]  # 0.5 assumes midpoint is in the middle of the normalized range

    # Fit the logistic curve with an increased number of maximum function evaluations
    popt, _ = curve_fit(logistic, x_data_normalized, y_data, p0=initial_guess, maxfev=5000)

    # Generate x values for the fitted curve on the normalized scale, then convert back to original scale
    x_fit_normalized = np.linspace(0, 1, 100)
    x_fit = x_fit_normalized * (x_max - x_min) + x_min
    y_fit = logistic(x_fit_normalized, *popt)

    # Plotting the data and the fit
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data Points')
    plt.plot(x_fit, y_fit, 'r-', label='Logistic Fit: L=%5.3f, k=%5.3f, x0=%5.3f' % tuple(popt))
    plt.title(f"Saturation Curve for {channel_name}")
    plt.xlabel('Media Spend')
    plt.ylabel('Contributions')
    plt.legend()
    plt.grid(True)
    plt.show()
