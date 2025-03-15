import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import List, Dict, Optional, Union, Callable
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

import folium
import geopandas as gpd
from typing import List, Dict, Optional, Union, Callable, Any
import json

def load_data(file):
    """Load data from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def get_numeric_columns(df):
    """Get list of numeric columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_datetime_columns(df):
    """Get list of datetime columns"""
    return df.select_dtypes(include=['datetime64']).columns.tolist()

def create_line_chart(df, x_column, y_column, title, template="plotly"):
    """Create line chart using plotly"""
    fig = px.line(df, x=x_column, y=y_column, title=title, template=template)
    fig.update_layout(
        title_x=0.5,
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=True,
        hovermode='x unified'
    )
    fig.update_traces(line_width=2)
    return fig

def create_bar_chart(df, x_column, y_column, title, template="plotly"):
    """Create bar chart using plotly"""
    fig = px.bar(df, x=x_column, y=y_column, title=title, template=template)
    fig.update_layout(
        title_x=0.5,
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=True,
        bargap=0.2
    )
    return fig

def create_scatter_plot(df, x_column, y_column, title, template="plotly"):
    """Create scatter plot using plotly"""
    fig = px.scatter(df, x=x_column, y=y_column, title=title, template=template)
    fig.update_layout(
        title_x=0.5,
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=True
    )
    fig.update_traces(marker=dict(size=8))
    return fig

def create_pie_chart(df, column, title, template="plotly"):
    """Create pie chart using plotly"""
    value_counts = df[column].value_counts()
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=title,
        template=template
    )
    fig.update_layout(
        title_x=0.5,
        showlegend=True
    )
    return fig

def create_correlation_matrix(df: pd.DataFrame, template="plotly"):
    """Create correlation matrix heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title="Correlation Matrix",
        title_x=0.5,
        template=template,
        width=800,
        height=800
    )
    return fig

def create_box_plot(df: pd.DataFrame, numeric_column: str, group_by: Optional[str] = None, template="plotly"):
    """Create box plot for numeric column"""
    fig = px.box(df, y=numeric_column, x=group_by, title=f"Box Plot: {numeric_column}", template=template)
    fig.update_layout(title_x=0.5)
    return fig

def create_violin_plot(df: pd.DataFrame, numeric_column: str, group_by: Optional[str] = None, template="plotly"):
    """Create violin plot for numeric column"""
    fig = px.violin(df, y=numeric_column, x=group_by, title=f"Violin Plot: {numeric_column}", template=template)
    fig.update_layout(title_x=0.5)
    return fig

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """Handle missing values in the dataset"""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    return df

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """Handle outliers in specified columns"""
    df_clean = df.copy()

    for column in columns:
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)

    return df_clean

def calculate_summary_stats(df: pd.DataFrame, column: str) -> Dict:
    """Calculate detailed summary statistics for a column"""
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    return stats

def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic summary statistics"""
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        return numeric_df.describe()
    return pd.DataFrame()

def create_time_series_plot(df: pd.DataFrame, date_column: str, value_column: str, 
                          title: str, window: int = 7, template="plotly"):
    """Create time series plot with moving average"""
    df_sorted = df.sort_values(date_column)
    df_sorted['MA'] = df_sorted[value_column].rolling(window=window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=df_sorted[value_column],
        name='Original',
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=df_sorted['MA'],
        name=f'{window}-point Moving Average',
        mode='lines',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title=title,
        title_x=0.5,
        template=template,
        xaxis_title=date_column,
        yaxis_title=value_column,
        showlegend=True
    )
    return fig


def perform_statistical_tests(df: pd.DataFrame, column1: str, column2: str = None) -> Dict:
    """Perform various statistical tests on the data"""
    results = {}

    # Normality test
    stat, p_value = stats.normaltest(df[column1].dropna())
    results['normality_test'] = {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }

    if column2:
        # Correlation test
        corr, p_value = stats.pearsonr(df[column1].dropna(), df[column2].dropna())
        results['correlation_test'] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # T-test
        t_stat, p_value = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
        results['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    return results

def create_probability_plot(df: pd.DataFrame, column: str, template="plotly") -> go.Figure:
    """Create a probability plot for distribution analysis"""
    data = df[column].dropna()
    fig = go.Figure()

    # Calculate probplot
    osm, osr = stats.probplot(data, dist="norm")

    fig.add_trace(go.Scatter(
        x=osm[0],
        y=osm[1],
        mode='markers',
        name='Data',
        marker=dict(color='blue')
    ))

    # Add the reference line
    fig.add_trace(go.Scatter(
        x=osm[0],
        y=osr[0] * osm[0] + osr[1],
        mode='lines',
        name='Reference Line',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f"Probability Plot for {column}",
        title_x=0.5,
        template=template,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        showlegend=True
    )

    return fig

def decompose_time_series(df: pd.DataFrame, date_column: str, value_column: str, 
                         period: int = 7, template="plotly") -> go.Figure:
    """Decompose time series into trend, seasonal, and residual components"""
    df_sorted = df.sort_values(date_column).copy()

    # Perform decomposition
    decomposition = seasonal_decompose(
        df_sorted[value_column],
        period=period,
        extrapolate_trend='freq'
    )

    # Create subplots
    fig = go.Figure()

    # Original
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=decomposition.observed,
        name='Original',
        line=dict(color='blue')
    ))

    # Trend
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=decomposition.trend,
        name='Trend',
        line=dict(color='red')
    ))

    # Seasonal
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=decomposition.seasonal,
        name='Seasonal',
        line=dict(color='green')
    ))

    # Residual
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=decomposition.resid,
        name='Residual',
        line=dict(color='gray')
    ))

    fig.update_layout(
        title="Time Series Decomposition",
        title_x=0.5,
        template=template,
        height=800,
        showlegend=True
    )

    return fig

def analyze_feature_relationships(df: pd.DataFrame, 
                                target_column: str, 
                                feature_columns: List[str],
                                template="plotly") -> Dict[str, go.Figure]:
    """Analyze relationships between features and target variable"""
    figures = {}

    # Correlation heatmap
    correlation = df[feature_columns + [target_column]].corr()
    figures['correlation'] = go.Figure(data=go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu'
    ))
    figures['correlation'].update_layout(
        title="Feature Correlations",
        title_x=0.5,
        template=template
    )

    # Scatter plots for each feature
    for feature in feature_columns:
        if df[feature].dtype in ['int64', 'float64']:
            fig = px.scatter(
                df,
                x=feature,
                y=target_column,
                title=f"{feature} vs {target_column}",
                template=template
            )
            figures[feature] = fig

    return figures

def generate_data_profile(df: pd.DataFrame) -> Dict:
    """Generate comprehensive data profile"""
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum()
        },
        'column_info': {}
    }

    for column in df.columns:
        col_info = {
            'dtype': str(df[column].dtype),
            'missing_values': df[column].isnull().sum(),
            'missing_percentage': (df[column].isnull().sum() / len(df)) * 100
        }

        if df[column].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'skew': df[column].skew(),
                'kurtosis': df[column].kurtosis(),
                'unique_values': df[column].nunique()
            })
        else:
            col_info.update({
                'unique_values': df[column].nunique(),
                'most_common': df[column].value_counts().head(5).to_dict()
            })

        profile['column_info'][column] = col_info

    return profile

def create_forecast(df: pd.DataFrame, date_column: str, value_column: str, 
                   periods: int = 30, seasonal_periods: int = 7) -> pd.DataFrame:
    """Create simple forecast using Holt-Winters method"""
    df_sorted = df.sort_values(date_column).copy()
    model = ExponentialSmoothing(
        df_sorted[value_column],
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add'
    )
    fitted_model = model.fit()

    # Generate future dates
    last_date = pd.to_datetime(df_sorted[date_column].iloc[-1])
    future_dates = pd.date_range(
        start=last_date,
        periods=periods + 1,
        freq='D'
    )[1:]  # Exclude the last known date

    # Make forecast
    forecast = fitted_model.forecast(periods)
    forecast_df = pd.DataFrame({
        date_column: future_dates,
        value_column: forecast,
        'type': 'forecast'
    })

    # Add type column to original data
    df_sorted['type'] = 'actual'

    return pd.concat([df_sorted, forecast_df])

def apply_custom_aggregation(df: pd.DataFrame, group_by: str, 
                           agg_column: str, agg_func: Union[str, Callable]) -> pd.DataFrame:
    """Apply custom aggregation to the data"""
    if callable(agg_func):
        return df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    return df.groupby(group_by)[agg_column].agg(agg_func).reset_index()

def export_data(df: pd.DataFrame, format: str = 'csv') -> Union[str, bytes]:
    """Export data in various formats"""
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'excel':
        return df.to_excel(index=False)
    elif format == 'json':
        return df.to_json(orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_custom_theme(primary_color: str, secondary_color: str, 
                       background_color: str) -> Dict:
    """Create custom theme for charts"""
    return {
        'layout': {
            'plot_bgcolor': background_color,
            'paper_bgcolor': background_color,
            'font': {'color': primary_color},
            'title': {'font': {'color': primary_color}},
            'xaxis': {
                'gridcolor': secondary_color,
                'linecolor': secondary_color,
                'tickcolor': secondary_color,
            },
            'yaxis': {
                'gridcolor': secondary_color,
                'linecolor': secondary_color,
                'tickcolor': secondary_color,
            }
        }
    }

def add_chart_annotations(fig: go.Figure, annotations: List[Dict]) -> go.Figure:
    """Add custom annotations to a chart"""
    for annotation in annotations:
        fig.add_annotation(
            x=annotation.get('x'),
            y=annotation.get('y'),
            text=annotation.get('text', ''),
            showarrow=annotation.get('show_arrow', True),
            arrowhead=annotation.get('arrow_head', 1),
            ax=annotation.get('ax', 0),
            ay=annotation.get('ay', -40)
        )
    return fig

def create_advanced_visualization(df: pd.DataFrame, 
                                chart_type: str,
                                x_column: str,
                                y_column: str,
                                title: str,
                                custom_theme: Optional[Dict] = None,
                                annotations: Optional[List[Dict]] = None,
                                **kwargs) -> go.Figure:
    """Create advanced visualization with custom styling and annotations"""
    if chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column, title=title, **kwargs)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_column, y=y_column, title=title, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=title, **kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # Apply custom theme if provided
    if custom_theme:
        fig.update_layout(custom_theme)

    # Add annotations if provided
    if annotations:
        fig = add_chart_annotations(fig, annotations)

    return fig

def create_map_visualization(df: pd.DataFrame, 
                           lat_column: str, 
                           lon_column: str,
                           value_column: Optional[str] = None,
                           zoom_start: int = 2) -> folium.Map:
    """Create an interactive map visualization"""
    # Calculate center point
    center_lat = df[lat_column].mean()
    center_lon = df[lon_column].mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Add markers
    for idx, row in df.iterrows():
        popup_text = f"{value_column}: {row[value_column]}" if value_column else "Location"
        folium.CircleMarker(
            location=[row[lat_column], row[lon_column]],
            radius=8,
            popup=popup_text,
            fill=True
        ).add_to(m)

    return m

def add_chart_annotation(fig: go.Figure, 
                        annotations: List[Dict[str, Any]], 
                        author: str,
                        timestamp: Optional[str] = None) -> go.Figure:
    """Add collaborative annotations to a chart"""
    for annotation in annotations:
        fig.add_annotation(
            x=annotation.get('x'),
            y=annotation.get('y'),
            text=f"{annotation.get('text', '')}\n- {author}",
            showarrow=True,
            arrowhead=1,
            ax=annotation.get('ax', 0),
            ay=annotation.get('ay', -40),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10)
        )

        if timestamp:
            # Add timestamp at the bottom of annotation
            fig.add_annotation(
                x=annotation.get('x'),
                y=annotation.get('y'),
                text=f"Added: {timestamp}",
                showarrow=False,
                yshift=-30,
                font=dict(size=8, color="gray")
            )

    return fig

def save_annotations(annotations: List[Dict[str, Any]], 
                    chart_id: str,
                    file_path: str = "annotations.json") -> None:
    """Save annotations to a JSON file"""
    try:
        # Load existing annotations
        try:
            with open(file_path, 'r') as f:
                all_annotations = json.load(f)
        except FileNotFoundError:
            all_annotations = {}

        # Update annotations for this chart
        all_annotations[chart_id] = annotations

        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)

    except Exception as e:
        raise Exception(f"Error saving annotations: {str(e)}")

def load_annotations(chart_id: str, 
                    file_path: str = "annotations.json") -> List[Dict[str, Any]]:
    """Load annotations for a specific chart"""
    try:
        with open(file_path, 'r') as f:
            all_annotations = json.load(f)
            return all_annotations.get(chart_id, [])
    except FileNotFoundError:
        return []
    except Exception as e:
        raise Exception(f"Error loading annotations: {str(e)}")