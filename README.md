
# Data Visualization Dashboard

https://abetworks-data-visualizer.streamlit.app/

An interactive data visualization dashboard built with Streamlit that provides comprehensive data analysis and visualization capabilities.

## Features

- Data Management
- Interactive Visualizations
- Advanced Analytics
- Geospatial Analysis
- Custom Theme Settings
- Export Options

## Requirements

This project requires Python 3.11+ and the following packages:
- folium>=0.19.5
- geopandas>=1.0.1
- numpy>=2.2.3
- openpyxl>=3.1.5
- pandas>=2.2.3
- plotly>=6.0.0
- scipy>=1.15.2
- seaborn>=0.13.2
- statsmodels>=0.14.4
- streamlit>=1.43.2

## Deployment Instructions

1. Fork this project on Replit
2. Click on "Deploy" in the workspace header
3. Choose "Autoscale" deployment type
4. Configure deployment settings:
   - Run command: `streamlit run app.py --server.port 5000`
   - Port: 5000 (external port: 80)
5. Click "Deploy" to publish your application

## Local Development

To run the dashboard locally on Replit:
1. Click the "Run" button
2. The app will be available at the URL shown in the console
3. Upload a CSV or Excel file to begin analyzing data

## Project Structure

- `app.py`: Main application file
- `utils.py`: Utility functions
- `attached_assets/`: Static assets
- `.streamlit/`: Streamlit configuration
