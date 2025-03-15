import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import (
    load_data, get_numeric_columns, get_datetime_columns,
    create_line_chart, create_bar_chart, create_scatter_plot, create_pie_chart,
    create_correlation_matrix, create_box_plot, create_violin_plot,
    handle_missing_values, handle_outliers, calculate_summary_stats,
    generate_summary_stats, create_time_series_plot, create_forecast, 
    create_advanced_visualization, apply_custom_aggregation, create_custom_theme,
    perform_statistical_tests, create_probability_plot,
    decompose_time_series, analyze_feature_relationships,
    generate_data_profile, create_map_visualization,
    add_chart_annotation, save_annotations, load_annotations
)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Data Visualization Dashboard",
        layout="wide"
    )

    # Add company logo in sidebar
    st.sidebar.image("attached_assets/aBETwORLKS.png", width=200)
    st.sidebar.markdown("---")

    # Main title with styling
    st.title("Interactive Data Visualization Dashboard")
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # File upload
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)

            # Data Management Section
            st.header("Data Management")
            data_management_tab, visualization_tab, advanced_analysis_tab, settings_tab, geospatial_tab = st.tabs([
                "Data Management", "Visualization", "Advanced Analysis", "Settings", "Geospatial Analysis"
            ])

            with data_management_tab:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Data Cleaning")
                    missing_strategy = st.selectbox(
                        "Handle Missing Values",
                        ["None", "drop", "mean", "median", "mode"]
                    )

                    if missing_strategy != "None":
                        df = handle_missing_values(df, missing_strategy)

                    numeric_cols = get_numeric_columns(df)
                    if numeric_cols:
                        st.subheader("Handle Outliers")
                        outlier_cols = st.multiselect(
                            "Select columns for outlier treatment",
                            numeric_cols
                        )
                        if outlier_cols:
                            outlier_method = st.selectbox(
                                "Outlier handling method",
                                ["None", "iqr"]
                            )
                            if outlier_method != "None":
                                df = handle_outliers(df, outlier_cols, outlier_method)

                with col2:
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    st.text(f"Total Rows: {len(df)}")
                    st.text(f"Total Columns: {len(df.columns)}")

                    # Download cleaned data
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Cleaned Data",
                        csv,
                        "cleaned_data.csv",
                        "text/csv",
                        key='download-cleaned-csv'
                    )

            with visualization_tab:
                st.header("Create Visualizations")

                # Sidebar controls
                st.sidebar.header("Visualization Controls")

                # Get column lists
                numeric_columns = get_numeric_columns(df)
                all_columns = df.columns.tolist()
                datetime_columns = get_datetime_columns(df)

                # Chart type selection with custom styling
                chart_type = st.sidebar.selectbox(
                    "Select Chart Type",
                    ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Box Plot", "Violin Plot"]
                )

                # Chart customization options
                st.sidebar.subheader("Chart Customization")
                custom_title = st.sidebar.text_input("Chart Title", "")
                chart_theme = st.sidebar.selectbox(
                    "Chart Theme",
                    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
                )

                if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_column = st.selectbox("Select X-axis", all_columns)
                    with col2:
                        y_column = st.selectbox("Select Y-axis", numeric_columns)

                    title = custom_title if custom_title else f"{chart_type}: {y_column} vs {x_column}"

                    if chart_type == "Line Chart":
                        fig = create_line_chart(df, x_column, y_column, title, chart_theme)
                    elif chart_type == "Bar Chart":
                        fig = create_bar_chart(df, x_column, y_column, title, chart_theme)
                    else:  # Scatter Plot
                        fig = create_scatter_plot(df, x_column, y_column, title, chart_theme)

                    st.plotly_chart(fig, use_container_width=True)

                    # Add annotation features
                    st.subheader("Add Annotations")
                    annotation_x = st.text_input("X-coordinate for annotation")
                    annotation_y = st.text_input("Y-coordinate for annotation")
                    annotation_text = st.text_area("Annotation text")
                    author = st.text_input("Your name")

                    if st.button("Add Annotation"):
                        try:
                            chart_id = f"{chart_type}_{x_column}_{y_column}"
                            annotations = load_annotations(chart_id)
                            new_annotation = {
                                'x': float(annotation_x),
                                'y': float(annotation_y),
                                'text': annotation_text
                            }
                            annotations.append(new_annotation)
                            save_annotations(annotations, chart_id)

                            # Add annotation to chart
                            fig = add_chart_annotation(
                                fig, 
                                [new_annotation], 
                                author,
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("Annotation added successfully!")
                        except Exception as e:
                            st.error(f"Error adding annotation: {str(e)}")

                elif chart_type in ["Box Plot", "Violin Plot"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        numeric_col = st.selectbox("Select Numeric Column", numeric_columns)
                    with col2:
                        group_by = st.selectbox("Group By (optional)", ["None"] + all_columns)

                    group_by = None if group_by == "None" else group_by
                    title = custom_title if custom_title else f"{chart_type}: {numeric_col}"

                    if chart_type == "Box Plot":
                        fig = create_box_plot(df, numeric_col, group_by, chart_theme)
                    else:  # Violin Plot
                        fig = create_violin_plot(df, numeric_col, group_by, chart_theme)

                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pie Chart":
                    column = st.selectbox("Select Column", all_columns)
                    title = custom_title if custom_title else f"Pie Chart: {column} Distribution"
                    fig = create_pie_chart(df, column, title, chart_theme)
                    st.plotly_chart(fig, use_container_width=True)

            with advanced_analysis_tab:
                st.header("Advanced Analysis")

                # Correlation Analysis
                if numeric_columns:
                    st.subheader("Correlation Analysis")
                    corr_fig = create_correlation_matrix(df, chart_theme)
                    st.plotly_chart(corr_fig, use_container_width=True, key='correlation_matrix')

                # Time Series Analysis
                if datetime_columns:
                    st.subheader("Time Series Analysis")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        date_col = st.selectbox("Select Date Column", datetime_columns)
                    with col2:
                        value_col = st.selectbox("Select Value Column", numeric_columns)
                    with col3:
                        window = st.slider("Moving Average Window", 2, 30, 7)

                    title = f"Time Series Analysis: {value_col} over {date_col}"
                    ts_fig = create_time_series_plot(df, date_col, value_col, title, window, chart_theme)
                    st.plotly_chart(ts_fig, use_container_width=True, key='time_series_plot')

                # Predictive Analytics
                if datetime_columns:
                    st.subheader("Predictive Analytics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        forecast_date = st.selectbox("Select Date Column for Forecast", datetime_columns)
                    with col2:
                        forecast_value = st.selectbox("Select Value Column for Forecast", numeric_columns)
                    with col3:
                        forecast_periods = st.slider("Forecast Periods", 7, 90, 30)

                    if st.button("Generate Forecast"):
                        try:
                            forecast_df = create_forecast(df, forecast_date, forecast_value, forecast_periods)
                            fig = create_advanced_visualization(
                                forecast_df,
                                "line",
                                forecast_date,
                                forecast_value,
                                "Forecast Analysis",
                                color="type"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")

                # Custom Aggregation
                st.subheader("Custom Aggregation")
                col1, col2, col3 = st.columns(3)
                with col1:
                    group_column = st.selectbox("Group By", all_columns)
                with col2:
                    agg_column = st.selectbox("Column to Aggregate", numeric_columns)
                with col3:
                    agg_function = st.selectbox("Aggregation Function", 
                        ["mean", "sum", "min", "max", "count", "median"])

                if st.button("Calculate Aggregation"):
                    try:
                        agg_df = apply_custom_aggregation(df, group_column, agg_column, agg_function)
                        st.write("Aggregation Results:")
                        st.dataframe(agg_df)

                        # Visualize aggregation
                        fig = create_bar_chart(agg_df, group_column, agg_column, 
                            f"{agg_function.capitalize()} of {agg_column} by {group_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error performing aggregation: {str(e)}")

                # Statistical Summary
                st.subheader("Statistical Summary")
                if numeric_columns:
                    selected_col = st.selectbox("Select Column for Statistics", numeric_columns)
                    stats = calculate_summary_stats(df, selected_col)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Basic Statistics:")
                        st.write(pd.DataFrame([stats]))

                    with col2:
                        st.write("Distribution Plot:")
                        fig = create_box_plot(df, selected_col, template=chart_theme)
                        st.plotly_chart(fig, use_container_width=True, key=f'box_plot_{selected_col}')

                # Advanced Statistical Analysis
                st.subheader("Advanced Statistical Analysis")
                if numeric_columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        stat_col1 = st.selectbox("Select First Column", numeric_columns, key='stat_col1')
                    with col2:
                        stat_col2 = st.selectbox("Select Second Column (Optional)", 
                                               ["None"] + numeric_columns, key='stat_col2')

                    stat_col2 = None if stat_col2 == "None" else stat_col2

                    if st.button("Run Statistical Tests"):
                        try:
                            results = perform_statistical_tests(df, stat_col1, stat_col2)

                            # Display results
                            st.write("### Test Results")

                            # Normality Test
                            st.write("#### Normality Test")
                            norm_result = results['normality_test']
                            st.write(f"P-value: {norm_result['p_value']:.4f}")
                            st.write(f"Data is {'normally' if norm_result['is_normal'] else 'not normally'} distributed")

                            if stat_col2:
                                # Correlation Test
                                st.write("#### Correlation Test")
                                corr_result = results['correlation_test']
                                st.write(f"Correlation: {corr_result['correlation']:.4f}")
                                st.write(f"P-value: {corr_result['p_value']:.4f}")
                                st.write(f"Correlation is {'significant' if corr_result['significant'] else 'not significant'}")

                                # T-Test
                                st.write("#### T-Test")
                                t_result = results['t_test']
                                st.write(f"T-statistic: {t_result['t_statistic']:.4f}")
                                st.write(f"P-value: {t_result['p_value']:.4f}")
                                st.write(f"Difference is {'significant' if t_result['significant'] else 'not significant'}")

                            # Probability Plot
                            st.write("### Probability Plot")
                            prob_fig = create_probability_plot(df, stat_col1, chart_theme)
                            st.plotly_chart(prob_fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error performing statistical tests: {str(e)}")

                # Time Series Decomposition
                if datetime_columns:
                    st.subheader("Time Series Decomposition")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        decomp_date = st.selectbox("Select Date Column", datetime_columns, key='decomp_date')
                    with col2:
                        decomp_value = st.selectbox("Select Value Column", numeric_columns, key='decomp_value')
                    with col3:
                        period = st.slider("Select Period", 2, 30, 7, key='decomp_period')

                    if st.button("Perform Decomposition"):
                        try:
                            decomp_fig = decompose_time_series(df, decomp_date, decomp_value, period, chart_theme)
                            st.plotly_chart(decomp_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error performing time series decomposition: {str(e)}")

                # Feature Relationships
                st.subheader("Feature Relationship Analysis")
                if numeric_columns:
                    target_col = st.selectbox("Select Target Variable", numeric_columns, key='target_col')
                    feature_cols = st.multiselect("Select Feature Columns", 
                                                [col for col in numeric_columns if col != target_col],
                                                key='feature_cols')

                    if feature_cols and st.button("Analyze Relationships"):
                        try:
                            relationship_figs = analyze_feature_relationships(
                                df, target_col, feature_cols, chart_theme
                            )

                            # Display correlation heatmap
                            st.write("#### Feature Correlations")
                            st.plotly_chart(relationship_figs['correlation'], use_container_width=True)

                            # Display scatter plots
                            st.write("#### Feature Scatter Plots")
                            for feature, fig in relationship_figs.items():
                                if feature != 'correlation':
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error analyzing feature relationships: {str(e)}")

                # Data Profiling
                st.subheader("Data Profiling")
                if st.button("Generate Data Profile"):
                    try:
                        profile = generate_data_profile(df)

                        # Display basic information
                        st.write("### Basic Information")
                        st.write(pd.DataFrame([profile['basic_info']]))

                        # Display column information
                        st.write("### Column Information")
                        for column, info in profile['column_info'].items():
                            with st.expander(f"Column: {column}"):
                                st.write(pd.DataFrame([info]))

                                if df[column].dtype in ['int64', 'float64']:
                                    fig = create_box_plot(df, column, template=chart_theme)
                                    st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating data profile: {str(e)}")

                # Advanced Filtering
                st.subheader("Advanced Filtering")
                filter_cols = st.multiselect("Select Columns for Filtering", all_columns)

                if filter_cols:
                    filtered_df = df.copy()
                    for col in filter_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            filter_range = st.slider(
                                f"Filter {col}",
                                min_val, max_val,
                                (min_val, max_val)
                            )
                            filtered_df = filtered_df[
                                (filtered_df[col] >= filter_range[0]) & 
                                (filtered_df[col] <= filter_range[1])
                            ]
                        else:
                            unique_vals = df[col].unique()
                            selected_vals = st.multiselect(
                                f"Select values for {col}",
                                unique_vals,
                                default=list(unique_vals)
                            )
                            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

                    st.write("Filtered Data Preview:")
                    st.dataframe(filtered_df)

                    # Download filtered data
                    filtered_csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "Download Filtered Data",
                        filtered_csv,
                        "filtered_data.csv",
                        "text/csv",
                        key='download-filtered-csv'
                    )

            with settings_tab:
                st.header("Dashboard Settings")

                # Theme Customization
                st.subheader("Chart Theme Customization")
                col1, col2, col3 = st.columns(3)
                with col1:
                    primary_color = st.color_picker("Primary Color", "#1f77b4")
                with col2:
                    secondary_color = st.color_picker("Secondary Color", "#777777")
                with col3:
                    background_color = st.color_picker("Background Color", "#ffffff")

                # Create custom theme
                custom_theme = create_custom_theme(primary_color, secondary_color, background_color)

                # Export Options
                st.subheader("Export Options")
                export_format = st.selectbox("Export Format", ["csv", "excel", "json"])

                if st.button("Export Data"):
                    try:
                        if export_format == "csv":
                            export_data = df.to_csv(index=False)
                            file_name = "data.csv"
                            mime = "text/csv"
                        elif export_format == "excel":
                            export_data = df.to_excel(index=False)
                            file_name = "data.xlsx"
                            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        else:  # json
                            export_data = df.to_json(orient='records')
                            file_name = "data.json"
                            mime = "application/json"

                        st.download_button(
                            f"Download as {export_format.upper()}",
                            export_data,
                            file_name,
                            mime,
                            key=f'download-{export_format}'
                        )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")

                # Save/Load Settings
                st.subheader("Save/Load Preferences")
                if st.button("Save Current Settings"):
                    st.session_state.custom_theme = custom_theme
                    st.success("Settings saved!")

            with geospatial_tab:
                st.header("Geospatial Analysis")
                try:
                    # Check for latitude and longitude columns
                    lat_col = st.selectbox(
                        "Select Latitude Column",
                        [col for col in df.columns if 'lat' in col.lower()],
                        key='lat_col'
                    )
                    lon_col = st.selectbox(
                        "Select Longitude Column",
                        [col for col in df.columns if 'lon' in col.lower()],
                        key='lon_col'
                    )

                    value_col = st.selectbox(
                        "Select Value Column (Optional)",
                        ["None"] + get_numeric_columns(df),
                        key='value_col'
                    )
                    value_col = None if value_col == "None" else value_col

                    if st.button("Create Map"):
                        try:
                            map_obj = create_map_visualization(
                                df, lat_col, lon_col, value_col
                            )
                            # Convert to HTML string
                            map_html = map_obj._repr_html_()
                            st.components.v1.html(map_html, height=600)
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                except Exception as e:
                    st.error(f"Error in geospatial analysis: {str(e)}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file to begin")

if __name__ == "__main__":
    main()