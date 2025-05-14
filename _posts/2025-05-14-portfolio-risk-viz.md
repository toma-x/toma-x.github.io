---
layout: post
title: Real-time Portfolio Risk Visualizer
---

## Real-time Portfolio Risk Visualizer: A Journey into FastAPI and Plotly Dash

This semester, alongside my coursework, I embarked on a personal project that pushed my understanding of backend development, financial risk metrics, and interactive data visualization. The goal was to build a Real-time Portfolio Risk Visualizer. The idea originated from a desire to apply some of the quantitative finance concepts I’d been learning in a more tangible way.

The core of the project involved two main components: a FastAPI backend responsible for live Value at Risk (VaR) calculations and stress testing, and a Plotly Dash frontend to provide an interactive interface for dynamic portfolio monitoring.

### The Backend: FastAPI for Speed and Simplicity

I chose FastAPI for the backend primarily due to its advertised performance and the ease of use with Python's type hints. I’d read some comparisons with Flask and Django, and for an API-focused project that needed to be relatively lightweight and quick to develop, FastAPI seemed like a good fit. The automatic OpenAPI documentation was also a huge plus, something I knew would save me time down the line.

My initial hurdle was figuring out how to structure the FastAPI application. I started with a single `main.py` file, but it quickly became apparent that for modularity, especially with different risk calculations planned, I needed a more organized approach. I ended up structuring it with separate routers for different functionalities like portfolio management and risk calculations.

One of the first things I tackled was the VaR calculation. I decided to implement a Monte Carlo VaR. I knew historical simulation VaR was simpler, but I wanted to explore a more flexible method that could potentially incorporate different distributions later.

Getting the Monte Carlo simulation right took a fair bit of trial and error. Initially, my simulations were incredibly slow. I was generating random numbers for each asset in a loop, which, for a portfolio of even a moderate size over thousands of simulations, was a bottleneck.

I remember spending a good chunk of an afternoon on StackOverflow and reading through NumPy documentation. The breakthrough came when I realized I could vectorize the random number generation for all assets and all simulations at once using `np.random.multivariate_normal`, assuming a certain covariance structure. This dramatically improved performance.

Here’s a snippet of what the VaR calculation endpoint started to look like:

```python
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.stats import norm

router = APIRouter()

# Placeholder for where I'd load or define portfolio data
# In a real scenario, this would come from a DB or user input
current_portfolio_data = {
    "AAPL": {"weight": 0.4, "mu": 0.001, "sigma": 0.02},
    "MSFT": {"weight": 0.3, "mu": 0.0008, "sigma": 0.018},
    "GOOGL": {"weight": 0.3, "mu": 0.0009, "sigma": 0.019}
}

class PortfolioInput(BaseModel):
    assets: dict # Expecting something like {"AAPL": 10000, "MSFT": 8000}
    confidence_level: float = 0.99
    horizon_days: int = 1
    num_simulations: int = 10000

@router.post("/calculate_var_mc")
async def calculate_var_monte_carlo(portfolio_input: PortfolioInput):
    try:
        asset_names = list(portfolio_input.assets.keys())
        asset_values = np.array(list(portfolio_input.assets.values()))
        total_portfolio_value = np.sum(asset_values)

        if total_portfolio_value == 0:
            raise HTTPException(status_code=400, detail="Portfolio value cannot be zero.")

        weights = asset_values / total_portfolio_value
        
        # These would ideally come from a more robust source or calculation
        mus = np.array([current_portfolio_data[asset]["mu"] for asset in asset_names])
        sigmas = np.array([current_portfolio_data[asset]["sigma"] for asset in asset_names])

        # Simplified covariance matrix - assuming independence for this early version
        # This was a major simplification I made due to time constraints.
        # A proper implementation would use historical data to estimate this.
        cov_matrix = np.diag(sigmas**2) 

        # Adjust mus for the horizon
        daily_mus = mus * portfolio_input.horizon_days
        
        # Simulate returns
        # The `allow_singular=True` was something I added after getting errors with certain test covariance matrices.
        # I found a forum post suggesting this, though I need to investigate the implications more thoroughly.
        simulated_returns = np.random.multivariate_normal(daily_mus, cov_matrix * portfolio_input.horizon_days, 
                                                       portfolio_input.num_simulations)
        
        portfolio_sim_returns = np.dot(simulated_returns, weights)
        
        # Calculate VaR
        var_value = np.percentile(portfolio_sim_returns, (1 - portfolio_input.confidence_level) * 100)
        var_amount = total_portfolio_value * var_value

        return {"var_percent": var_value, "var_amount": var_amount, "confidence_level": portfolio_input.confidence_level}

    except Exception as e:
        # Basic error handling, this definitely needs to be more robust.
        raise HTTPException(status_code=500, detail=str(e))

```

The stress testing endpoint was conceptually simpler to start with. I defined a few scenarios – like a market crash (e.g., all assets drop by X%) or specific sector shocks. The challenge here was more about defining meaningful scenarios rather than the coding itself. Initially, I just hardcoded them, but the plan is to allow users to define their own scenarios via the frontend.

### The Frontend: Plotly Dash for Interactivity

For the frontend, I chose Plotly Dash. I have some experience with Python, and Dash allows you to build web applications purely in Python, which was appealing as it meant I didn't have to dive deep into JavaScript frameworks for this project. The tight integration with Plotly for charting was perfect for a visualizer.

The learning curve with Dash was a bit steeper than I anticipated, especially around callbacks and state management. My first attempt at a dynamic graph involved a lot of frustrating moments where the graph wouldn't update, or would update with the wrong data.

One particular issue I remember vividly was trying to chain callbacks. I wanted an input field to update a data store, which would then trigger multiple graphs to re-render. The Dash documentation on `dcc.Store` was my best friend here. I also found a few helpful community forum posts that clarified how to use `Input` and `Output` components effectively for more complex interactions.

Here’s a very simplified example of how I started structuring the Dash app layout and a basic callback for updating a graph. This is a barebones version just to illustrate the concept:

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import requests # To call the FastAPI backend

# This would be the URL of my running FastAPI backend
API_URL = "http://127.0.0.1:8000" 

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-time Portfolio Risk Visualizer"),
    
    html.Div([
        dcc.Input(id='asset-input-aapl', type='number', value=10000, placeholder="AAPL Value"),
        dcc.Input(id='asset-input-msft', type='number', value=8000, placeholder="MSFT Value"),
        dcc.Input(id='asset-input-googl', type='number', value=7000, placeholder="GOOGL Value"),
        html.Button('Calculate VaR', id='calculate-var-button', n_clicks=0)
    ]),
    
    html.Div(id='var-output-container'),
    
    dcc.Graph(id='portfolio-pie-chart')
])

@app.callback(
    Output('var-output-container', 'children'),
    Output('portfolio-pie-chart', 'figure'),
    Input('calculate-var-button', 'n_clicks'),
    State('asset-input-aapl', 'value'),
    State('asset-input-msft', 'value'),
    State('asset-input-googl', 'value'),
    prevent_initial_call=True # Avoid callback firing on page load
)
def update_var_and_pie(n_clicks, val_aapl, val_msft, val_googl):
    if n_clicks > 0:
        portfolio_payload = {
            "assets": {
                "AAPL": val_aapl if val_aapl else 0,
                "MSFT": val_msft if val_msft else 0,
                "GOOGL": val_googl if val_googl else 0
            },
            "confidence_level": 0.99, # Default for now
            "horizon_days": 1,
            "num_simulations": 5000 # Fewer sims for quicker frontend response in dev
        }
        
        var_text = "Error fetching VaR."
        pie_fig = px.pie(names=['N/A'], values=, title="Portfolio Allocation") # Default empty pie

        try:
            response = requests.post(f"{API_URL}/risk/calculate_var_mc", json=portfolio_payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            var_data = response.json()
            var_text = f"Calculated VaR ({var_data['confidence_level']*100}% CL): {var_data['var_amount']:.2f} ({var_data['var_percent']*100:.2f}%)"

            # Update pie chart
            df_portfolio = pd.DataFrame({
                "Asset": list(portfolio_payload["assets"].keys()),
                "Value": list(portfolio_payload["assets"].values())
            })
            df_portfolio = df_portfolio[df_portfolio["Value"] > 0] # Only show assets with value
            if not df_portfolio.empty:
                 pie_fig = px.pie(df_portfolio, values='Value', names='Asset', title='Portfolio Allocation')
            
        except requests.exceptions.RequestException as e:
            var_text = f"API Connection Error: {e}"
        except Exception as e:
            var_text = f"An error occurred: {e}"
            
        return var_text, pie_fig
    return dash.no_update, dash.no_update # If button not clicked, do nothing

```

A significant challenge was making the application feel "real-time." Initially, every small change in input parameters would trigger a full recalculation on the backend, which, even with the vectorized NumPy operations, could introduce a slight lag with a high number of simulations. I considered WebSockets for a more persistent connection, but for the scope of this project, I decided to stick with HTTP requests and focused on optimizing the calculation speed and using Dash's `prevent_initial_call` and careful state management to avoid unnecessary callbacks. I also debounced some inputs on the frontend eventually, though that's not shown in the simplified snippet above, to prevent firing off API requests too frequently while a user is still typing.

### Connecting Frontend and Backend

Ensuring the frontend and backend communicated smoothly was crucial. I used the `requests` library in my Dash callbacks to send data to the FastAPI endpoints and retrieve the results. Debugging this link often involved having both the FastAPI server running in one terminal (with `uvicorn main:app --reload`) and the Dash app in another, and meticulously checking the logs and network requests in the browser's developer tools. Misaligned Pydantic models on the FastAPI side with the JSON payload sent from Dash caused a few `422 Unprocessable Entity` errors that took some time to track down. It usually boiled down to a naming mismatch or incorrect data type.

### Future Work and Reflections

This project has been an incredible learning experience. There are many areas I’d like to improve. The VaR calculation currently uses a simplified covariance matrix; integrating historical data to calculate a more realistic covariance matrix is a key next step. Expanding the stress testing module to allow user-defined scenarios and more complex macroeconomic variable shocks would also be valuable.

On the FastAPI side, I need to implement proper authentication and more robust error handling and logging. For the Dash frontend, improving the UI/UX, adding more visualization types (like historical performance charts or VaR backtesting displays), and perhaps exploring ways to make the "real-time" aspect even smoother (maybe with `dcc.Interval` for periodic updates or delving into WebSockets if the complexity is justified) are on the list.

The process of building this, from conceptualizing the risk calculations to wrestling with Dash callbacks, has been challenging but immensely rewarding. It’s one thing to learn about these technologies and financial concepts in isolation, but integrating them into a working application provides a much deeper understanding. I also learned the importance of iterative development – starting small, getting something working, and then building upon it. Trying to build everything perfectly from the start would have been overwhelming.