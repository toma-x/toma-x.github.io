---
layout: post
title: Payroll Variance Anomaly Detector
---

## Detecting the Undetected: Building a Payroll Variance Anomaly Detector

Payroll is one of those critical functions in any organization, and errors or unusual fluctuations can be a huge headache, especially when it comes to audits. For a while now, I've been looking for a project that would let me dive deeper into practical machine learning, and the idea of an automated system to flag potential payroll anomalies seemed like a challenging yet rewarding endeavor. So, I decided to build a Python tool to do just that. My goal was to leverage Pandas for data wrangling and scikit-learn for the anomaly detection part, hoping to create something that could efficiently identify outliers in payroll datasets.

The core problem I wanted to tackle was identifying significant variances in payroll figures that might indicate errors, data entry mistakes, or even potentially fraudulent activity. Manually combing through payroll registers, especially for larger companies, is incredibly time-consuming and prone to human error. An automated system could serve as a first-pass filter, highlighting records that warrant closer inspection.

Initially, I thought about a purely rule-based system – for instance, flagging any payment that's, say, 50% higher than the previous period for the same employee. But payroll is complex. A legitimate bonus, a promotion, or a significant amount of overtime could all cause such a spike. A purely rule-based system would either be too simplistic and miss subtle anomalies or become overly complex and hard to maintain. This led me towards machine learning, specifically unsupervised anomaly detection, where the model learns what "normal" looks like from the data itself.

Python was a no-brainer for this. The data science ecosystem is just so rich. Pandas is indispensable for any kind of data manipulation, and scikit-learn is the standard for ML tasks. My first step was to get a handle on the kind of data I'd be working with. I simulated a dataset structure that I thought would be reasonably representative, including fields like `EmployeeID`, `PayPeriodEndDate`, `GrossPay`, `RegularHours`, `OvertimeHours`, `Department`, `JobRole`, and importantly, `PreviousGrossPay` to help calculate variances.

Data preprocessing, as always, took up a significant chunk of time. The first hurdle was handling missing data. For `PreviousGrossPay`, a missing value for a new employee is expected. I decided to impute these with the median `GrossPay` of their `JobRole` for their first recorded pay period, assuming their first pay is more likely to be "normal" for that role. For other missing numerical values, like `OvertimeHours` (if it was `NaN` instead of 0), I opted to fill with 0 after confirming this was the most logical approach for this dataset.

One crucial part was feature engineering. Simply feeding raw pay amounts into an anomaly detector wouldn't capture the "variance" aspect effectively. I created a few new features:
*   `PayVarianceAbsolute`: The absolute difference between `GrossPay` and `PreviousGrossPay`.
*   `PayVariancePercentage`: The percentage change from `PreviousGrossPay` to `GrossPay`. This one felt more robust to scale differences between high earners and low earners.
*   `DeviationFromRoleMedianPay`: How much an employee's `GrossPay` deviates from the median `GrossPay` of their `JobRole` in the current pay period.

Here’s a rough idea of how I calculated the percentage variance using Pandas:

```python
# Assuming 'df' is my pandas DataFrame
# Ensure PreviousGrossPay is not zero to avoid division by zero errors
df['PreviousGrossPay'] = df['PreviousGrossPay'].replace(0, np.nan) # Temporarily mark zeros as NaN
# Forward fill NaN PreviousGrossPay for consecutive periods if an employee had a zero pay then a normal one
df['PreviousGrossPay'] = df.groupby('EmployeeID')['PreviousGrossPay'].ffill()

# For the actual calculation, ensure PreviousGrossPay is not zero
# If PreviousGrossPay is still NaN (e.g. new employee), variance might be high or handled differently
# For simplicity here, let's say we fill remaining NaNs in PreviousGrossPay with GrossPay to make variance zero for first period
df['PreviousGrossPay'].fillna(df['GrossPay'], inplace=True)

# Now calculate, being careful about division by zero if PreviousGrossPay was actually zero and not NaN
df['PayVariancePercentage'] = np.where(df['PreviousGrossPay'] != 0,
                                     ((df['GrossPay'] - df['PreviousGrossPay']) / df['PreviousGrossPay']) * 100,
                                     0) # Assign 0% variance if previous was 0 and current is also 0, or handle as extreme if current is non-zero

# For employees with 0 previous pay and non-zero current pay, this could be flagged as an anomaly by its sheer value,
# or one might assign a very high percentage (e.g. by replacing 0 with a very small number for division).
# I decided to cap it or handle it via the absolute difference feature as well.
```
Getting this logic right, especially handling new employees (where `PreviousGrossPay` would be null) or employees returning after a break, took a few iterations. I remember spending a good hour debugging why some variances were showing up as `inf` or `NaN` before realizing I had to be more careful with zero values in `PreviousGrossPay` and the order of operations for new hires.

For the anomaly detection model, I looked into a few options in scikit-learn. Local Outlier Factor (LOF) and One-Class SVM were contenders. LOF seemed powerful but I read on a few forums (I think it was a StackOverflow discussion on anomaly detection for transactional data) that it could be computationally intensive for larger datasets, which was a concern. One-Class SVM felt a bit like a black box to me at the time, and I wanted to understand the mechanics better. I eventually settled on `IsolationForest`. It’s generally good at spotting outliers, tends to be efficient, and the underlying principle – that anomalies are "few and different" and thus easier to isolate in a tree structure – made intuitive sense to me.

When implementing the `IsolationForest`, the main hyperparameter I focused on was `contamination`. This parameter tells the model the expected proportion of outliers in the dataset. This was tricky because, in a real-world scenario, you often don't know this proportion upfront. I started with a common default, like `0.01` or `0.05` (1% or 5% anomalies), and planned to iterate. I also kept `n_estimators` (the number of trees) at the default of 100 initially, as increasing it too much would slow down training, and I wanted to get a baseline first.

Here’s a snippet of how I set up and trained the model, focusing on the engineered features:

```python
from sklearn.ensemble import IsolationForest
# ... other imports like pandas, numpy

# Features I decided to use for detection
# These were selected after some trial and error; initially, I had too many.
features_for_detection = ['GrossPay', 'PayVariancePercentage', 'DeviationFromRoleMedianPay', 'OvertimeHours']
X_train = df[features_for_detection].copy()

# Handle potential NaNs or Infs in the selected features, maybe from variance calculation
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# I used median imputation for any remaining NaNs in the training features
for col in X_train.columns:
    if X_train[col].isnull().any():
        X_train[col].fillna(X_train[col].median(), inplace=True)

# Initialize and fit the model
# The 'auto' setting for contamination was not available in the version I was using initially,
# so I had to set it manually. For newer versions, 'auto' is an option.
# Let's assume I found 0.03 to be a reasonable starting point after some experimentation
model_iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
model_iso_forest.fit(X_train)

# Predict outliers: -1 for outliers, 1 for inliers
df['anomaly_score'] = model_iso_forest.decision_function(X_train)
df['is_anomaly_iso_forest'] = model_iso_forest.predict(X_train)
```
One "aha!" moment came when I realized how sensitive the `PayVariancePercentage` could be. An employee going from a $1 temporary pay correction back to their normal $1000 salary would show a massive percentage increase. While an anomaly, it wasn't the kind of subtle error I was also looking for. This reinforced the need to use multiple features, including absolute changes and deviations from role medians, rather than relying on a single variance metric. I also considered feature scaling (like `StandardScaler`), but since Isolation Forest is tree-based, it's not strictly required. I did try it, but it didn't make a noticeable difference in my initial results, so I skipped it to keep the preprocessing simpler.

Evaluating the model was the next big step. Anomaly detection is tricky because you often lack perfectly labeled data. I created a small, manually curated test set where I knew certain records were genuine anomalies (either based on plausible error scenarios I invented or by slightly tweaking existing records to make them anomalous). On this test set, I was trying to see how many of my known anomalies were caught (recall) and how many of the items flagged by the model were actual anomalies (precision).

Achieving the "95% accuracy" I mentioned earlier was more nuanced. It wasn't a simple accuracy score from a confusion matrix in the typical classification sense, as the "normal" class is overwhelmingly large. Instead, this figure primarily reflected the precision on the flagged anomalies in my test set combined with qualitative feedback. I considered a detection "accurate" if, upon review of the top N anomalies flagged by the model (ranked by their anomaly score), about 95% of them were indeed records that merited investigation according to my predefined criteria for the test set. It took a fair bit of tuning the `contamination` parameter and tweaking the feature set to get to a point where the flagged items were mostly meaningful. For example, if `contamination` was too high, it flagged too many borderline cases. Too low, and it missed obvious ones. I found that a value around 0.02-0.04 seemed to work best for my synthetic dataset.

One specific struggle was interpreting the results. The model would flag a record, but then I had to dig into *why*. Having the `anomaly_score` helped, as more negative scores indicated stronger outliers. I would sort by this score and then manually inspect the features of the top flagged records to see if they made sense. This iterative process of training, predicting, inspecting, and then adjusting features or model parameters was key. For instance, I noticed initially it was flagging many high overtime values. While potentially anomalous, some were legitimate. I then had to consider if `OvertimeHours` alone was too noisy a feature, or if its interaction with `JobRole` (e.g., some roles never have overtime) was what mattered.

Looking back, this project was a fantastic learning experience. Wrestling with imperfect data, engineering relevant features, and iteratively tuning a model to get meaningful results taught me a lot more than just running library functions. There were definitely moments of frustration, like when my variance calculations kept producing weird edge cases, or when I couldn't figure out why the model was flagging seemingly normal records until I re-evaluated my feature set. One particular bug that cost me an evening was a simple Pandas indexing mistake where I was accidentally creating a new column instead of modifying an existing one during a conditional update, leading to skewed feature values for a subset of data. I only found it by painstakingly printing out dataframe states at multiple steps.

There’s still plenty of room for improvement. I could explore other algorithms like LOF more deeply, or even ensemble methods. A more sophisticated approach to defining "normal" for an employee, perhaps by looking at their own historical pay trajectory using time series methods, could be a next step. And, of course, building a simple interface (maybe with Streamlit or Flask) for an end-user to upload data and review flagged anomalies would make it much more practical.

Despite the challenges, getting the anomaly detector to a point where it could reliably flag suspicious payroll entries with decent accuracy felt like a real achievement. It underscored how powerful these tools can be when applied thoughtfully to real-world problems.