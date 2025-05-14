---
layout: post
title: Portfolio Optimization Visualizer
---

## Portfolio Optimization Visualizer: A Deep Dive into the Build

This semester, alongside my usual coursework, I dedicated a good chunk of time to a personal project that I've been wanting to tackle for a while: a Portfolio Optimization Visualizer. The idea was to build a tool that could take inputs for various assets and then compute and display the efficient frontier based on mean-variance optimization. It was quite a journey, especially integrating a Python backend with an Angular frontend, and I wanted to document the process, the hurdles, and the small victories.

### The Core Idea and Technology Choices

The goal was to create an interactive application. Users should be able to define a set of financial assets, specifying their expected returns and the covariance matrix representing their risks and interdependencies. The application would then calculate the set of optimal portfolios – the efficient frontier – and visualize it.

For the backend, Python felt like a natural choice due to its strong presence in finance and data science. I opted for **FastAPI** because I'd heard good things about its speed, ease of use, and automatic documentation generation. For the actual optimization, **cvxpy** seemed like the right tool. It’s a powerful Python library for convex optimization problems, and mean-variance optimization fits that description perfectly.

On the frontend, I wanted to use a modern framework. I’ve been meaning to get more experience with **Angular**, and with version 17 being relatively new, it felt like a good opportunity to work with its latest features. The plan was to build an interactive UI where users could input their asset data and see the efficient frontier plotted on a chart.

### Backend: FastAPI and the cvxpy Engine

Setting up the FastAPI backend was surprisingly straightforward. The documentation is excellent. My main endpoint needed to accept the asset data: expected returns, the covariance matrix, and the number of assets.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cvxpy as cp

app = FastAPI()

class PortfolioData(BaseModel):
    expected_returns: list[float]
    covariance_matrix: list[list[float]]
    risk_free_rate: float = 0.0 # Default to 0 if not provided

# ... (CORS middleware setup would go here)
```

The real meat of the backend is the optimization logic. Given `n` assets, the expected returns vector `mu` (an n-vector), and the covariance matrix `Sigma` (an n x n matrix), the Markowitz mean-variance optimization problem for a target portfolio return `R_target` can be formulated as minimizing portfolio variance `w.T * Sigma * w` subject to `w.T * mu = R_target` and `sum(w) = 1`, where `w` is the vector of asset weights. Non-negativity constraints (`w >= 0`) are also common to prevent short selling.

I decided to trace out the efficient frontier by iterating through a range of possible target returns. For each target return, I'd solve the optimization problem to find the portfolio with the minimum variance.

Here’s a snippet of how I approached the `cvxpy` part:

```python
def calculate_efficient_frontier(returns_np, cov_matrix_np, num_portfolios=50):
    num_assets = len(returns_np)
    weights = cp.Variable(num_assets)
    target_return = cp.Parameter(nonneg=True) # Parameter for the target return

    # Objective: Minimize portfolio variance
    risk = cp.quad_form(weights, cov_matrix_np)
    objective = cp.Minimize(risk)

    # Constraints
    constraints = [
        cp.sum(weights) == 1,           # Sum of weights is 1
        weights >= 0,                   # No short selling
        returns_np.T @ weights == target_return # Target portfolio return
    ]

    problem = cp.Problem(objective, constraints)

    # Iterate through target returns to trace the frontier
    frontier_returns = np.linspace(returns_np.min(), returns_np.max(), num_portfolios)
    frontier_risks = []
    frontier_weights = []

    for r_target_val in frontier_returns:
        target_return.value = r_target_val
        try:
            problem.solve(solver=cp.ECOS) # Using ECOS solver, could also try SCS
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                frontier_risks.append(np.sqrt(risk.value))
                frontier_weights.append(weights.value.tolist())
            else:
                # If not optimal, append NaN or handle appropriately
                # This was a tricky part, sometimes solvers wouldn't converge
                # for extreme target returns, especially if constraints were tight.
                frontier_risks.append(np.nan) 
                # frontier_weights.append([np.nan] * num_assets) # placeholder for weights
        except cp.SolverError:
            # Handle cases where the solver itself fails
            frontier_risks.append(np.nan)
            # frontier_weights.append([np.nan] * num_assets)


    # We need to return actual returns achieved, not just target ones for plotting
    # For now, I am returning the target returns used for calculation along with their corresponding risks
    # and the calculated weights.
    # A more robust solution might re-calculate the actual portfolio return from optimal weights.
    return {"risks": frontier_risks, "returns": frontier_returns.tolist(), "weights": frontier_weights}

@app.post("/optimize")
async def optimize_portfolio(data: PortfolioData):
    try:
        returns_np = np.array(data.expected_returns)
        cov_matrix_np = np.array(data.covariance_matrix)

        if returns_np.shape != cov_matrix_np.shape or returns_np.shape != cov_matrix_np.shape:
            raise HTTPException(status_code=400, detail="Dimension mismatch in returns or covariance matrix.")
        if returns_np.ndim != 1:
            raise HTTPException(status_code=400, detail="Expected returns should be a 1D array.")

        # Basic validation for covariance matrix symmetry and positive definiteness could be added
        # For now, trusting the input a bit for my own tool.

        results = calculate_efficient_frontier(returns_np, cov_matrix_np)
        # Filter out NaN results before sending back
        # This was important as sometimes the solver wouldn't find an optimal solution
        # for all target returns, especially at the extremes.
        valid_indices = [i for i, r in enumerate(results["risks"]) if not np.isnan(r)]
        
        filtered_risks = [results["risks"][i] for i in valid_indices]
        filtered_returns = [results["returns"][i] for i in valid_indices]
        filtered_weights = [results["weights"][i] for i in valid_indices if i < len(results["weights"])]


        if not filtered_risks: # if all were NaN
            raise HTTPException(status_code=500, detail="Optimization failed to produce valid points for the efficient frontier.")

        return {"risks": filtered_risks, "returns": filtered_returns, "weights": filtered_weights}

    except cp.SolverError as e:
        # This was a common one when first setting up cvxpy or if data was problematic
        raise HTTPException(status_code=500, detail=f"Solver error during optimization: {str(e)}")
    except Exception as e:
        # General catch-all for other unexpected issues
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

```

One early struggle was getting `cvxpy` to work with different solvers. `ECOS` is generally good for these types of problems, but I experimented with `SCS` as well. Sometimes, for certain input data, one solver would perform better or converge when another wouldn't. Another tricky bit was handling cases where the optimization problem is infeasible for a given target return. This typically happens if you ask for a return higher than what any individual asset offers, or lower than the minimum variance portfolio if you can't short. My initial attempts often resulted in the backend crashing or `cvxpy` throwing errors that were hard to decipher. I eventually settled on catching these issues and returning `NaN` values, which the frontend then had to handle by not plotting those points.

I also spent a fair bit of time on StackOverflow looking up `cvxpy` syntax for specific constraints, especially when I was trying to add more complex ones like group constraints (which I eventually decided against for this version to keep things manageable). The `cvxpy` documentation is comprehensive but can be dense, so seeing practical examples in forums was key.

A crucial part was setting up CORS (Cross-Origin Resource Sharing) middleware in FastAPI. Since my Angular frontend would be running on `localhost:4200` and FastAPI on `localhost:8000`, browsers would block requests by default. FastAPI’s `CORSMiddleware` made this relatively painless after I figured out which origins and methods to allow.

```python
# In the main FastAPI app setup, after app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:4200", # Angular app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)
```
This felt like a rite of passage for anyone building a separate frontend and backend. Forgetting this initially led to a lot of confusing "network error" messages in the browser console.

### Frontend: Angular 17 for Interaction and Visualization

With the backend taking shape, I turned to Angular 17. My goal was a simple UI: input fields for the number of assets, and then dynamically generated input fields for each asset's expected return and the covariance matrix.

I used Angular's reactive forms for handling the inputs. This was a bit of a learning curve, especially for the dynamic generation of form controls for the covariance matrix.

```typescript
// Inside an Angular component, e.g., portfolio-form.component.ts
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, FormArray, Validators, FormControl } from '@angular/forms';
import { PortfolioService } from '../portfolio.service'; // My service for API calls

interface PlotPoint {
  x: number; // Risk
  y: number; // Return
}

interface ChartData {
  datasets: { data: PlotPoint[], label: string, borderColor?: string, fill?: boolean }[];
}

@Component({
  selector: 'app-portfolio-form',
  templateUrl: './portfolio-form.component.html',
  // styleUrls: ['./portfolio-form.component.css'] // My styles
})
export class PortfolioFormComponent implements OnInit {
  portfolioForm: FormGroup;
  numAssets: number = 2; // Default
  efficientFrontierData: ChartData | null = null;
  errorMessage: string | null = null;
  isLoading: boolean = false;

  // Chart.js configuration
  public lineChartOptions: any = { // Using 'any' here for simplicity, ideally type this
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Risk (Standard Deviation)' } },
      y: { title: { display: true, text: 'Expected Return' } }
    }
  };
  public lineChartLegend = true;
  public lineChartType = 'scatter'; // Scatter plot for efficient frontier


  constructor(private fb: FormBuilder, private portfolioService: PortfolioService) {
    this.portfolioForm = this.fb.group({
      numAssetsInput: [this.numAssets, [Validators.required, Validators.min(2)]],
      expectedReturns: this.fb.array([]),
      covarianceMatrixRows: this.fb.array([])
    });
  }

  ngOnInit(): void {
    this.updateFormArrays(this.numAssets);
    this.portfolioForm.get('numAssetsInput')?.valueChanges.subscribe(val => {
      if (this.portfolioForm.get('numAssetsInput')?.valid) {
        this.numAssets = val;
        this.updateFormArrays(val);
      }
    });
  }

  updateFormArrays(num: number): void {
    // Update expected returns
    const returnsArray = this.portfolioForm.get('expectedReturns') as FormArray;
    while (returnsArray.length !== num) {
      if (returnsArray.length < num) {
        returnsArray.push(this.fb.control(0.1, Validators.required)); // Default value
      } else {
        returnsArray.removeAt(returnsArray.length - 1);
      }
    }

    // Update covariance matrix
    const matrixRows = this.portfolioForm.get('covarianceMatrixRows') as FormArray;
    while (matrixRows.length !== num) {
      if (matrixRows.length < num) {
        matrixRows.push(this.fb.array([]));
      } else {
        matrixRows.removeAt(matrixRows.length - 1);
      }
    }

    matrixRows.controls.forEach((row, i) => {
      const rowArray = row as FormArray;
      while (rowArray.length !== num) {
        if (rowArray.length < num) {
          // Initialize diagonal with a small positive variance, off-diagonal with small covariance
          let defaultValue = (rowArray.length === i) ? 0.05 : 0.01;
          rowArray.push(this.fb.control(defaultValue, Validators.required));
        } else {
          rowArray.removeAt(rowArray.length - 1);
        }
      }
    });
  }

  get expectedReturnsControls() {
    return (this.portfolioForm.get('expectedReturns') as FormArray).controls;
  }

  getCovarianceMatrixRowsControls() {
    return (this.portfolioForm.get('covarianceMatrixRows') as FormArray).controls;
  }

  getCovarianceRowControls(rowIndex: number) {
    const matrixRows = this.portfolioForm.get('covarianceMatrixRows') as FormArray;
    return (matrixRows.at(rowIndex) as FormArray).controls;
  }

  onSubmit(): void {
    if (this.portfolioForm.valid) {
      this.isLoading = true;
      this.errorMessage = null;
      this.efficientFrontierData = null;

      const formValue = this.portfolioForm.value;
      const payload = {
        expected_returns: formValue.expectedReturns,
        covariance_matrix: formValue.covarianceMatrixRows,
        risk_free_rate: 0.02 // Could add an input for this later
      };

      this.portfolioService.getEfficientFrontier(payload).subscribe({
        next: (response) => {
          // Convert backend response to Chart.js format
          const points: PlotPoint[] = response.risks.map((risk, index) => ({
            x: risk,
            y: response.returns[index]
          }));
          this.efficientFrontierData = {
            datasets: [{ data: points, label: 'Efficient Frontier', borderColor: '#3e95cd', fill: false }]
          };
          this.isLoading = false;
        },
        error: (err) => {
          console.error("Error fetching efficient frontier:", err);
          this.errorMessage = err.error?.detail || "Failed to calculate efficient frontier. Check inputs or backend.";
          this.isLoading = false;
        }
      });
    } else {
        this.errorMessage = "Form is invalid. Please check your inputs.";
    }
  }
}
```
The `updateFormArrays` function was a bit fiddly to get right, ensuring the forms for returns and the covariance matrix dynamically adjusted to the specified number of assets. Making the covariance matrix symmetric by default or enforcing symmetry via validation was something I considered, but for this version, I relied on user input being correct. A future improvement would be to only ask for the upper (or lower) triangle and construct the full matrix.

For visualizing the efficient frontier, I chose **Chart.js** with the **ngx-charts-wrapper** (or just `ng2-charts` which is the more common Angular wrapper). It's relatively lightweight and easy to get started with. Integrating it was mostly about formatting the data from my FastAPI backend into the structure Chart.js expects (an array of `{x, y}` points).

```typescript
// portfolio.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

interface BackendResponse {
  risks: number[];
  returns: number[];
  weights: number[][]; // Optional: if you want to use weights on the frontend
}

@Injectable({
  providedIn: 'root'
})
export class PortfolioService {
  private apiUrl = 'http://localhost:8000/optimize'; // My FastAPI backend URL

  constructor(private http: HttpClient) { }

  getEfficientFrontier(data: any): Observable<BackendResponse> {
    return this.http.post<BackendResponse>(this.apiUrl, data)
      .pipe(
        catchError(this.handleError)
      );
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'Unknown error!';
    if (error.error instanceof ErrorEvent) {
      // Client-side errors
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side errors
      // The backend now sends a 'detail' field in its HTTPExceptions
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.error?.detail || error.message}`;
    }
    // console.error(errorMessage); // Logging it to console
    return throwError(() => new Error(errorMessage)); // Pass the error object or a custom error
  }
}
```
Error handling in the `PortfolioService` using `catchError` from RxJS was important. The backend can return various HTTP error codes, and displaying a meaningful message to the user rather than just a failed request in the console makes a big difference. Initially, I just had a generic error message, but then I updated it to try and parse the `detail` field from FastAPI's `HTTPException` responses.

One particular "gotcha" was dealing with Angular's change detection when updating the chart data. If I just pushed new data into the existing array, Chart.js wouldn't always pick up the changes. Reassigning the `efficientFrontierData` object entirely, as shown in the `onSubmit` method, forces Angular to detect the change and re-render the chart. This took a bit of debugging and reading through Angular and Chart.js forums to understand why my chart wasn't updating.

Styling the forms and ensuring responsiveness was also a time sink. I used basic CSS, aiming for clarity over flashy design. Getting the dynamically generated covariance matrix inputs to line up nicely in a grid took some trial and error with CSS flexbox/grid.

### Reflections and Future Thoughts

This project was a fantastic learning experience. Integrating a Python backend with an Angular frontend, while common in the industry, was new territory for me on a project of this scale. Debugging `cvxpy` issues, understanding the nuances of FastAPI request/response models, and wrestling with Angular's reactive forms and change detection were all challenging but ultimately rewarding.

There were moments of definite frustration. For instance, when `cvxpy` would give an "infeasible" or "unbounded" status without much context, I had to meticulously check my matrix math and constraints, often by simplifying the problem to just two assets and working my way up. Another time, I spent hours debugging why my Angular `HttpClient` calls weren't even reaching the backend, only to realize I had a typo in the API URL in my Angular service – a classic mistake.

If I were to continue this project, I'd add:
*   More robust input validation on both frontend and backend.
*   Allowing users to upload historical price data and calculating returns/covariance from that.
*   Displaying the portfolio weights for a selected point on the efficient frontier.
*   Plotting the Capital Market Line (CML) if a risk-free asset is considered.
*   More sophisticated optimization models (e.g., Black-Litterman, or adding constraints like maximum allocation per asset).
*   User accounts and saving portfolios.

Overall, I'm pretty pleased with how it turned out as a personal learning project. It solidified my understanding of portfolio theory and gave me practical experience with a full-stack application using these specific technologies. There's still a lot to refine, but it's a solid foundation.