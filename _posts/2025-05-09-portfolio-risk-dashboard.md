---
layout: post
title: Real-Time Portfolio Risk Dashboard
---

## Real-Time Portfolio Risk Dashboard: From Python and Angular to a Linux Box

After the AI-Powered Financial News Analyzer, I wanted to build something more interactive, something that could give a "live" feel to data. I’d been dabbling with some small investments and the idea of a real-time risk dashboard seemed like a natural next step. The goal was to have a web app where I could see some basic risk metrics for a hypothetical portfolio, updated as notionally new "trade" data came in. This involved a Python Flask API for the backend logic, an Angular UI for the front-end, PostgreSQL to store trade data, and then the fun part: deploying it all on my own Linux server using Docker.

**The Stack Choice: Familiarity and a Bit of a Challenge**

*   **Python with Flask:** Python is my go-to, and Flask is great for quickly spinning up APIs. I didn't need the kitchen-sink approach of Django for this. I just needed a few endpoints to serve portfolio data and maybe simulate adding new trades.
*   **Angular:** I’d used Angular in a group project for a class and, while the learning curve was a bit steep initially (TypeScript, RxJS, components, services – it’s a whole ecosystem!), I liked its structured approach. I wanted to get better at it, and a dynamic dashboard felt like a good use case.
*   **PostgreSQL:** I've used MySQL more in the past, but I'd heard good things about PostgreSQL's robustness and feature set, especially for data integrity. Since I was dealing with "financial" (albeit simulated) data, it felt like a more "serious" choice. [17]
*   **Docker:** My experience with Docker on the Vertex AI project was a real eye-opener. The ability to package everything up and run it consistently anywhere seemed like magic, and I knew I wanted to use it for this project, especially for deploying to my own server. [15, 12]
*   **Linux Server:** I have an old desktop tower that I’d previously turned into a basic Ubuntu server for tinkering. It was mostly gathering dust, so this was a perfect excuse to make it do some real work.

**Designing the Backend: Flask and Basic Risk Calcs**

The Flask API was relatively straightforward. I needed:
1.  Endpoints to get current portfolio holdings.
2.  An endpoint to add a new trade.
3.  An endpoint to retrieve calculated risk metrics.

For risk, I kept it simple to start: portfolio volatility (standard deviation of returns) and maybe a basic Value at Risk (VaR) calculation. I wasn't aiming for sophisticated financial modeling, just something indicative. [2, 5, 7]

My initial `app.py` looked something like this:

```python
# app.py (early version)
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS # Needed this pretty quickly
import numpy as np
import pandas as pd # For any data manipulation if needed

app = Flask(__name__)
CORS(app) # Allow all origins for local dev, learned this from last project and Auth0 guide

# --- Database Configuration ---
# Initially, I hardcoded this, which is bad, I know.
# Moved to environment variables later when Dockerizing.
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/portfolio_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

# --- Models ---
class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"<Trade {self.ticker} {self.quantity}@{self.price}>"

# db.create_all() # I quickly learned to use Flask-Migrate instead of calling this directly.

# --- API Endpoints ---
@app.route('/trades', methods=['POST'])
def add_trade():
    data = request.get_json()
    if not data or not all(k in data for k in ('ticker', 'quantity', 'price')):
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        new_trade = Trade(ticker=data['ticker'], quantity=int(data['quantity']), price=float(data['price']))
        db.session.add(new_trade)
        db.session.commit()
        return jsonify({'message': 'Trade added', 'id': new_trade.id}), 201
    except Exception as e:
        db.session.rollback()
        # app.logger.error(f"Error adding trade: {e}") # Added proper logging later
        return jsonify({'error': str(e)}), 500

@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    # This was a very naive initial implementation. 
    # It just sums up quantities per ticker.
    trades = Trade.query.all()
    portfolio = {}
    for trade in trades:
        if trade.ticker not in portfolio:
            portfolio[trade.ticker] = {'quantity': 0, 'total_cost': 0.0}
        portfolio[trade.ticker]['quantity'] += trade.quantity
        # This isn't really 'cost' but more like current value if prices were static
        portfolio[trade.ticker]['total_cost'] += trade.quantity * trade.price 
    return jsonify(portfolio)

# Placeholder for risk calculation
# I knew this would need more work, fetching historical prices, etc.
def calculate_portfolio_volatility(portfolio_data):
    # For now, just returning a dummy value
    # In a real scenario, I'd fetch historical prices for tickers in portfolio_data,
    # calculate daily returns, then the covariance matrix, and portfolio standard deviation.
    # mock_returns = np.random.rand(100, len(portfolio_data.keys())) * 0.01 # 100 days, N assets
    # daily_portfolio_returns = np.sum(mock_returns * (1/len(portfolio_data.keys())), axis=1) # Assuming equal weights for simplicity
    # volatility = np.std(daily_portfolio_returns) * np.sqrt(252) # Annualized
    return 0.15 # Placeholder 15% annualized volatility

@app.route('/risk', methods=['GET'])
def get_risk():
    # This endpoint would call the portfolio calculation and then risk metrics
    # portfolio_summary = get_portfolio_summary_logic() # a refactored version of get_portfolio
    # volatility = calculate_portfolio_volatility(portfolio_summary)
    # For now:
    volatility = calculate_portfolio_volatility({}) # Passing empty for now
    return jsonify({'annualized_volatility': volatility, 'var_95_1d': 0.02}) # Dummy VaR

if __name__ == '__main__':
    # IMPORTANT: For Docker, Flask's dev server needs to listen on 0.0.0.0
    # I spent a good hour debugging why my Angular app couldn't reach Flask in Docker
    # until I found a StackOverflow post explaining this.
    app.run(host='0.0.0.0', port=5000, debug=True)
```
One of the first hurdles was database migrations. Initially, I just had `db.create_all()` in my script. This works fine when you're starting, but as soon as I wanted to add a column or change a table, it was a pain. I remembered Flask-Migrate from some tutorials, which uses Alembic under the hood. Setting that up (`flask db init`, `flask db migrate`, `flask db upgrade`) made schema changes *so* much easier. [17]

Calculating actual risk metrics like volatility or VaR properly requires historical price data for the assets in the portfolio. [7, 8] I stubbed this out initially, planning to integrate a library like `yfinance` (like in my previous project) or a free market data API later. For the dashboard's first version, just showing *that* the risk metrics *would* appear was the main goal.

**Building the Angular Front-End: Services, Components, and RxJS Headaches**

Angular was where I spent a good chunk of my time. I set up a new project using the Angular CLI (`ng new portfolio-risk-ui`).

My basic structure involved:
*   A `TradeService` to handle HTTP calls to my Flask API.
*   A `PortfolioComponent` to display the current holdings.
*   A `RiskComponent` to display the risk metrics.
*   An `AddTradeFormComponent` to input new trades.

The `TradeService` was where I first wrestled with RxJS Observables more seriously.

```typescript
// trade.service.ts (simplified)
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, Subject } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

// Define simple interfaces for type safety
export interface Trade {
  id?: number;
  ticker: string;
  quantity: number;
  price: number;
  timestamp?: string;
}

export interface PortfolioPosition {
  [ticker: string]: {
    quantity: number;
    total_cost: number; // Or current_value, depending on backend logic
  };
}

export interface RiskMetrics {
  annualized_volatility: number;
  var_95_1d: number;
}

@Injectable({
  providedIn: 'root'
})
export class TradeService {
  private apiUrl = 'http://localhost:5000'; // This will change with Docker!

  // Subject to notify components when portfolio updates
  private portfolioUpdated = new Subject<void>();

  constructor(private http: HttpClient) { }

  addTrade(trade: Trade): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/trades`, trade)
      .pipe(
        tap(() => {
          this.portfolioUpdated.next(); // Notify subscribers
        }),
        catchError(this.handleError)
      );
  }

  getPortfolio(): Observable<PortfolioPosition> {
    return this.http.get<PortfolioPosition>(`${this.apiUrl}/portfolio`)
      .pipe(catchError(this.handleError));
  }

  getRiskMetrics(): Observable<RiskMetrics> {
    return this.http.get<RiskMetrics>(`${this.apiUrl}/risk`)
      .pipe(catchError(this.handleError));
  }

  // Observable for components to subscribe to for updates
  getPortfolioUpdateListener(): Observable<void> {
    return this.portfolioUpdated.asObservable();
  }

  private handleError(error: HttpErrorResponse) {
    // Basic error handling
    console.error(`Backend returned code ${error.status}, body was: `, error.error);
    return throwError(() => new Error('Something bad happened; please try again later.'));
  }
}
```
A key thing here was the `portfolioUpdated` Subject and `getPortfolioUpdateListener`. When a new trade is added successfully, I wanted the portfolio and risk components to automatically refresh. Using a Subject that components could subscribe to was the way I figured out to do this reactively.

Unsubscribing from Observables was another point of confusion. I read a bunch of articles and Stack Overflow posts about `takeUntil` and when `HttpClient` Observables complete automatically. [4, 6, 9, 11] For simple HTTP GETs that complete, it's often not strictly necessary to manually unsubscribe. [10] But for longer-lived subscriptions, or things like `ActivatedRoute` params, it is. I tried to use the `async` pipe in templates where possible to let Angular handle subscriptions. For manual subscriptions in components (like to my `portfolioUpdated` Subject), I made sure to implement `OnDestroy` and call `unsubscribe()`.

```typescript
// portfolio.component.ts (excerpt)
import { Component, OnInit, OnDestroy } from '@angular/core';
import { TradeService, PortfolioPosition } from '../trade.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-portfolio',
  templateUrl: './portfolio.component.html',
  styleUrls: ['./portfolio.component.css']
})
export class PortfolioComponent implements OnInit, OnDestroy {
  portfolio: PortfolioPosition = {};
  isLoading = false;
  error: string | null = null;
  private portfolioUpdateSubscription!: Subscription;

  constructor(private tradeService: TradeService) { }

  ngOnInit(): void {
    this.fetchPortfolio();
    this.portfolioUpdateSubscription = this.tradeService.getPortfolioUpdateListener().subscribe(() => {
      this.fetchPortfolio(); // Refetch when notified
    });
  }

  fetchPortfolio(): void {
    this.isLoading = true;
    this.tradeService.getPortfolio().subscribe({
      next: (data) => {
        this.portfolio = data;
        this.isLoading = false;
        this.error = null;
      },
      error: (err) => {
        this.error = err.message;
        this.isLoading = false;
        // console.error("Error fetching portfolio:", err); // more specific logging
      }
    });
  }

  // Essential to prevent memory leaks
  ngOnDestroy(): void {
    if (this.portfolioUpdateSubscription) {
      this.portfolioUpdateSubscription.unsubscribe();
    }
  }

  // Helper for template to iterate over portfolio object
  getPortfolioTickers(): string[] {
    return Object.keys(this.portfolio);
  }
}
```
Displaying the portfolio, which was an object with tickers as keys, in the Angular template required using `getPortfolioTickers()` and then iterating over that array with `*ngFor`.

**Dockerizing the Zoo: Flask, Angular, and PostgreSQL**

This was the part I was both excited and nervous about. I wanted three containers: one for Flask, one for Angular (served by Nginx), and one for PostgreSQL. Docker Compose was the tool to manage this. [12, 13]

*   **PostgreSQL Container:** This was the easiest. I just used the official `postgres` image from Docker Hub. [15] The key was setting environment variables for the user, password, and database, and mapping a volume to persist the data.

    ```yaml
    # docker-compose.yml (partial, for postgres)
    version: '3.8'
    services:
      postgres_db:
        image: postgres:13-alpine # Chose alpine for smaller size
        container_name: portfolio_postgres
        environment:
          POSTGRES_USER: myuser      # BAD - should use .env file
          POSTGRES_PASSWORD: mysecretpassword # BAD - should use .env file
          POSTGRES_DB: portfolio_db
        volumes:
          - postgres_data:/var/lib/postgresql/data
        ports:
          - "5432:5432" # Expose for local inspection if needed, but services should use Docker network
    # ...
    volumes:
      postgres_data: # Defines the named volume
    ```
    I quickly learned to put sensitive data like `POSTGRES_USER` and `POSTGRES_PASSWORD` into a `.env` file, which Docker Compose automatically picks up, rather than hardcoding them in `docker-compose.yml`. [17]

*   **Flask API Container:** This needed a `Dockerfile`.

    ```dockerfile
    # backend/Dockerfile
    FROM python:3.9-slim-buster # Using a slim image

    WORKDIR /app

    COPY requirements.txt requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    # This was crucial for Flask to be accessible from other containers
    # Default Flask dev server binds to 127.0.0.1 which isn't reachable from other containers.
    # Gunicorn is better for "production" inside Docker.
    # CMD ["python", "app.py"] 
    # Using Gunicorn instead:
    CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
    ```
    The `CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]` was a step up from just `python app.py`. Gunicorn is a proper WSGI server. And `0.0.0.0` is essential for it to be reachable within the Docker network.

*   **Angular UI Container (with Nginx):** This was a two-stage Docker build. First stage builds the Angular app, second stage copies the static files into an Nginx container. [22]

    ```dockerfile
    # frontend/Dockerfile
    # Stage 1: Build the Angular app
    FROM node:16-alpine as builder
    WORKDIR /app
    COPY package.json package-lock.json ./
    RUN npm install
    COPY . .
    # The --configuration production flag is important for optimizations
    RUN npm run build --configuration production 

    # Stage 2: Serve the built app with Nginx
    FROM nginx:1.21-alpine
    # Remove default Nginx welcome page
    RUN rm -rf /usr/share/nginx/html/*
    # Copy built assets from builder stage
    COPY --from=builder /app/dist/portfolio-risk-ui /usr/share/nginx/html
    # Copy custom Nginx config if needed (e.g., for proxying API requests)
    # COPY nginx.conf /etc/nginx/conf.d/default.conf 
    EXPOSE 80
    CMD ["nginx", "-g", "daemon off;"] # Keep Nginx in the foreground
    ```
    Initially, I had CORS issues when the Angular app (served by Nginx on port 80) tried to call the Flask API (on port 5000). While `flask-cors` helped during local dev, for a cleaner Docker setup, I configured Nginx to act as a reverse proxy for API calls. This meant the browser only ever talked to Nginx on port 80.

    My `nginx.conf` (simplified):
    ```nginx
    # frontend/nginx.conf (example)
    server {
        listen 80;
        server_name localhost; # Or your server's domain/IP

        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html; # Important for Angular routing
        }

        location /api/ { # All requests to /api/... will be proxied
            proxy_pass http://flask_api:5000/; # 'flask_api' is the service name in docker-compose
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
    ```
    This `proxy_pass http://flask_api:5000/` line was gold. `flask_api` is the name I gave to my Flask service in `docker-compose.yml`. Docker Compose handles the network resolution.

Putting it all together in `docker-compose.yml`:
```yaml
# docker-compose.yml (more complete)
version: '3.8'

services:
  flask_api:
    build: ./backend  # Path to the Flask app's Dockerfile directory
    container_name: portfolio_flask_api
    restart: unless-stopped
    environment:
      # Passed to Flask app, e.g. for database connection string
      DATABASE_URL: postgresql://myuser:mysecretpassword@postgres_db:5432/portfolio_db 
      FLASK_ENV: production # Or development
      # Any other env vars your app needs
    depends_on:
      - postgres_db # Wait for DB to be ready (though 'depends_on' doesn't guarantee DB service is fully initialized)
    # ports: # No need to expose Flask port directly to host if Nginx is proxying
    #   - "5000:5000" 
    networks:
      - portfolio_network

  angular_ui:
    build: ./frontend # Path to Angular app's Dockerfile directory
    container_name: portfolio_angular_ui
    restart: unless-stopped
    ports:
      - "80:80" # Expose Nginx to the host on port 80
    depends_on:
      - flask_api # Ensure API is up (conceptually, Nginx will just fail to proxy if not)
    networks:
      - portfolio_network

  postgres_db:
    image: postgres:13-alpine
    container_name: portfolio_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: myuser 
      POSTGRES_PASSWORD: mysecretpassword
      POSTGRES_DB: portfolio_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    # ports: # Not strictly needed to expose to host, containers connect over Docker network
    #   - "5432:5432" 
    networks:
      - portfolio_network

volumes:
  postgres_data:

networks:
  portfolio_network:
    driver: bridge
```
One tricky bit was the `DATABASE_URL` for Flask. Inside the Docker network, the Flask app connects to the PostgreSQL container using its service name (`postgres_db`) as the host, not `localhost`. [21] This took some trial and error with connection strings. Also, `depends_on` in Docker Compose doesn't wait for the database service to be *fully initialized and ready to accept connections*, just for the container to start. For more robust startup, I'd probably need to implement a wait script in my Flask app's entry point.

**Deployment on the Linux Server: The Moment of Truth**

Once `docker-compose build` and `docker-compose up -d` ran without errors locally, I moved to my Linux server. I copied the project files, installed Docker and Docker Compose on the server, and ran the same commands.

The biggest hiccup here was firewall configuration. My Ubuntu server had `ufw` (Uncomplicated Firewall) enabled. I had to explicitly allow traffic on port 80: `sudo ufw allow 80/tcp`. Without this, I could see the containers running (`docker ps`), but couldn't access the Angular app from my browser on another machine.

Another small win was setting `restart: unless-stopped` for the services in `docker-compose.yml`. This meant if the server rebooted, Docker would automatically try to bring my application containers back up. [15]

**Reflections and Next Steps**

This project felt like a significant step up in terms of managing a multi-component application. Docker and Docker Compose were the real heroes here, making the deployment process so much more manageable than if I'd tried to install and configure Nginx, Python, Node, and PostgreSQL directly on the server.

The dashboard is still basic. The risk calculations are placeholders. [20, 23, 24, 26, 27] Real-time data isn't *truly* real-time; it updates when I manually add a trade. Future improvements could involve:
*   Integrating a real market data feed for price updates.
*   Implementing more sophisticated risk metrics. [2, 5]
*   Using WebSockets for true real-time updates to the UI instead of polling or manual refresh triggers.
*   Adding user authentication.
*   More robust error handling and logging across all services.

But seeing the Angular UI load in my browser, served by Nginx, fetching data from Flask, which in turn queried PostgreSQL – all running in separate Docker containers on my own little server – was incredibly satisfying. It was a great learning experience in full-stack development and basic DevOps practices.