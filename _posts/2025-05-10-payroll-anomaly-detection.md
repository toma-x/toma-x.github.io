---
layout: post
title: Anomalous Payroll Data Detection
---

## Finding Needles in Haystacks: Detecting Anomalous Payroll Data with Python

After my last project on tax forms, I was keen to dive into another data-oriented challenge. This time, I wanted to explore anomaly detection, specifically within the context of payroll data. The idea of building a tool that could flag potentially erroneous or fraudulent payroll entries seemed both interesting and a good way to get more hands-on experience with some machine learning techniques. So, I embarked on creating an "Anomalous Payroll Data Detector."

The core of this project was to use Python, relying heavily on **Pandas** for data manipulation and **Scikit-learn** for the machine learning part. The goal was to train a model that could identify unusual patterns in a dataset of payroll records. Given that real payroll data is highly sensitive and not something I have access to, the first major step was to simulate a realistic dataset.

**Crafting the Data: A Simulation Exercise**

Generating believable (and usefully anomalous) payroll data was a mini-project in itself. I needed features that you'd typically find in a payroll system: Employee ID, Department, State, Pay Rate, Hours Worked, Overtime Hours, various Deduction types (like health, tax), and finally, Net Pay.

I used Python's `numpy` library for generating most of the numerical data, trying to create somewhat realistic distributions. For instance, pay rates were drawn from a normal distribution, but then I had to make sure they made sense for different departments. Hours worked were generally clustered around 40, with some overtime.

```python
import pandas as pd
import numpy as np

# Simulating some data
num_records = 5000
np.random.seed(42) # for reproducibility

departments = ['Engineering', 'Sales', 'HR', 'Marketing', 'Operations']
states = ['CA', 'TX', 'NY', 'FL', 'IL', 'GA', 'OH', 'NC', 'MI', 'NJ']

data = {
    'EmployeeID': range(1, num_records + 1),
    'Department': np.random.choice(departments, num_records),
    'State': np.random.choice(states, num_records),
    'BasePayRate': np.random.normal(loc=35, scale=10, size=num_records).clip(min=15, max=150), # Hourly rate
    'HoursWorked': np.random.normal(loc=40, scale=5, size=num_records).clip(min=10, max=80),
    'OvertimeHours': np.random.normal(loc=5, scale=5, size=num_records).clip(min=0, max=30)
}
payroll_df = pd.DataFrame(data)

# Some basic calculations for other fields
payroll_df['GrossPay'] = (payroll_df['BasePayRate'] * payroll_df['HoursWorked']) + \
                         (payroll_df['BasePayRate'] * 1.5 * payroll_df['OvertimeHours'])
payroll_df['TaxDeduction'] = payroll_df['GrossPay'] * np.random.normal(loc=0.15, scale=0.05, size=num_records).clip(min=0.05, max=0.3)
payroll_df['InsuranceDeduction'] = np.random.normal(loc=100, scale=30, size=num_records).clip(min=20, max=300)
payroll_df['NetPay'] = payroll_df['GrossPay'] - payroll_df['TaxDeduction'] - payroll_df['InsuranceDeduction']

# Introduce some anomalies
# This was an iterative process, trying to make them not *too* obvious but detectable
anomaly_indices = np.random.choice(payroll_df.index, size=int(num_records * 0.02), replace=False) # ~2% anomalies
for idx in anomaly_indices:
    anomaly_type = np.random.choice(['pay_rate', 'hours', 'deduction_error', 'state_pay_mismatch'])
    if anomaly_type == 'pay_rate':
        payroll_df.loc[idx, 'BasePayRate'] *= np.random.choice([5, 0.1, 10]) # Drastically higher or lower
    elif anomaly_type == 'hours':
        payroll_df.loc[idx, 'HoursWorked'] += np.random.choice([80, -30]) # Unusually high or low hours
        payroll_df.loc[idx, 'HoursWorked'] = payroll_df.loc[idx, 'HoursWorked'].clip(min=0, max=168) # cap at 168 hrs/week
    elif anomaly_type == 'deduction_error':
        payroll_df.loc[idx, 'TaxDeduction'] = payroll_df.loc[idx, 'GrossPay'] * np.random.choice([0.01, 0.7]) # Very low or high tax
    elif anomaly_type == 'state_pay_mismatch':
        # e.g. someone in a 'low-cost' state having a 'high-cost' state salary profile
        # This one was harder to simulate effectively without more complex rules
        if payroll_df.loc[idx, 'BasePayRate'] < 50 : # if they have a lower payrate
             payroll_df.loc[idx, 'BasePayRate'] *= 3 # make it higher
        else: # if they have a higher payrate
             payroll_df.loc[idx, 'BasePayRate'] /= 3 # make it lower
    # Recalculate dependent fields after introducing anomaly
    payroll_df.loc[idx, 'GrossPay'] = (payroll_df.loc[idx, 'BasePayRate'] * payroll_df.loc[idx, 'HoursWorked']) + \
                                      (payroll_df.loc[idx, 'BasePayRate'] * 1.5 * payroll_df.loc[idx, 'OvertimeHours'])
    payroll_df.loc[idx, 'TaxDeduction'] = payroll_df.loc[idx, 'GrossPay'] * np.random.normal(loc=0.15, scale=0.05).clip(min=0.05, max=0.3) # Re-calc tax too
    payroll_df.loc[idx, 'NetPay'] = payroll_df.loc[idx, 'GrossPay'] - payroll_df.loc[idx, 'TaxDeduction'] - payroll_df.loc[idx, 'InsuranceDeduction']

# print(payroll_df.head())
# print(f"Total records: {len(payroll_df)}, Anomalous records: {len(anomaly_indices)}")
```
Injecting anomalies was tricky. I didn't just want to put in garbage numbers; I tried to simulate errors like a misplaced decimal in a pay rate, or someone logging an impossible number of hours, or deductions being way off. I marked about 2% of my data as anomalous for training purposes, which felt like a reasonable starting point.

**Preprocessing: Getting the Data Model-Ready**

Before feeding this data into any model, some preprocessing was necessary. The 'Department' and 'State' columns were categorical. Machine learning models generally need numerical input. I considered Label Encoding first, but that implies an ordinal relationship (e.g., Sales < HR < Engineering), which isn't true for departments or states. So, I opted for One-Hot Encoding using `pd.get_dummies`. This creates new binary (0 or 1) columns for each category, which is a cleaner way to represent this type of data for many models.

```python
# Select features for the model
# Initially, I just threw almost everything in, then thought about what's most relevant
features_to_use = ['BasePayRate', 'HoursWorked', 'OvertimeHours', 'GrossPay', 'TaxDeduction', 'InsuranceDeduction', 'NetPay']
# Categorical features need to be handled
categorical_features = ['Department', 'State']

# One-Hot Encoding for categorical features
# I had to remember to concatenate these back to the main dataframe
payroll_processed_df = pd.get_dummies(payroll_df, columns=categorical_features, drop_first=True) # drop_first to reduce multicollinearity

# Update features_to_use with the new one-hot encoded column names
# This was a bit manual, had to check the new column names from get_dummies
encoded_dept_cols = [col for col in payroll_processed_df.columns if 'Department_' in col]
encoded_state_cols = [col for col in payroll_processed_df.columns if 'State_' in col]
final_features = features_to_use + encoded_dept_cols + encoded_state_cols

# I decided to keep EmployeeID out of the features for the model itself, but useful for identifying records later
X = payroll_processed_df[final_features]
# print(X.head())
# print(X.isnull().sum().sum()) # Checking for NaNs after processing, just in case.
```
I made sure there were no missing values, though with simulated data, this was less of an issue than in real-world scenarios. I also spent some time deciding which features to actually use for the model. My first instinct was to include everything, but then I refined it to fields that would logically show discrepancies.

**Choosing the Right Tool: Why Isolation Forest?**

For the anomaly detection model itself, I decided to use **Isolation Forest** from Scikit-learn. I'd read a few articles and forum posts (mostly on StackOverflow and Towards Data Science blogs) suggesting it was quite effective for this kind of task. Unlike distance-based methods (like k-Nearest Neighbors or some clustering approaches) which can struggle with high-dimensional data or require careful normalization, Isolation Forest works a bit differently. It builds an ensemble of "isolation trees" for the dataset. The idea is that anomalies, being "few and different," are easier to isolate (i.e., they tend to be closer to the root of these random trees).

I briefly considered other methods. Simple statistical approaches like Z-scores or Interquartile Range (IQR) are good for finding outliers in individual features, but I wanted something that could look at combinations of features. One-Class SVM was another option, but Isolation Forest seemed more computationally efficient and easier to get started with for this project. I remember reading that Isolation Forest doesn't strictly require feature scaling, which was a plus, though for some datasets, scaling might still be beneficial. I decided to try it without scaling first.

**Training the Model and the `contamination` Parameter**

Training the Isolation Forest was fairly straightforward with Scikit-learn. The main parameter I had to think about was `contamination`. This parameter tells the model the expected proportion of outliers in the dataset. Since I had intentionally introduced about 2% anomalies in my simulated data, I started with `contamination=0.02`.

```python
from sklearn.ensemble import IsolationForest

# Initialize and train the Isolation Forest model
# The 'auto' setting for contamination was an option, but I had an estimate
# I also played around with n_estimators
iso_forest_model = IsolationForest(n_estimators=100, # Number of trees
                                   contamination=0.02, # My estimate of anomalies
                                   random_state=42,  # For reproducibility
                                   # max_features=0.8, # Sometimes I tried limiting features per tree
                                  )

iso_forest_model.fit(X)

# Predict anomalies (-1 for outliers, 1 for inliers)
payroll_processed_df['anomaly_score'] = iso_forest_model.decision_function(X) # Lower scores are more anomalous
payroll_processed_df['is_anomaly_predicted'] = iso_forest_model.predict(X)

# print(payroll_processed_df[['EmployeeID', 'anomaly_score', 'is_anomaly_predicted']].head())
```
Getting the `contamination` value right felt like a bit of a guess. If set too high, it might flag too many normal records as anomalies (false positives). If too low, it might miss actual anomalies (false negatives). I experimented with values around my 2% mark. `n_estimators` (the number of trees) was another parameter; 100 seemed like a reasonable default to start, but I knew I could tune this if needed. I also made sure to set a `random_state` so my results would be reproducible each time I ran the script.

**Making Sense of the Anomalies**

Once the model was trained, I used its `predict()` method to flag records as either inliers (1) or outliers (-1). The `decision_function()` method was also useful as it gives an anomaly score for each data point – the lower the score, the more likely it is to be an anomaly.

```python
# Let's see what the model flagged as anomalous
detected_anomalies_df = payroll_processed_df[payroll_processed_df['is_anomaly_predicted'] == -1]

# print(f"Number of anomalies detected: {len(detected_anomalies_df)}")
# print("Details of detected anomalies:")
# For a real investigation, I'd look at the original (non-one-hot-encoded) features too for easier interpretation.
# This requires merging back or looking up by EmployeeID in the original payroll_df.
# For simplicity here, I'm showing from payroll_processed_df, but with original values where possible.

# To make the output more readable, let's join with original categorical values if possible
# This was a bit of a struggle to display nicely in the log
anomalies_to_review = payroll_df.loc[detected_anomalies_df.index]
# print(anomalies_to_review[['EmployeeID', 'BasePayRate', 'HoursWorked', 'GrossPay', 'NetPay', 'Department', 'State']])

# Example: One record that I knew I made anomalous
# record_100_info = payroll_df[payroll_df['EmployeeID'] == anomaly_indices+1] # +1 because EmployeeID starts at 1
# print("Known anomaly example (original):")
# print(record_100_info)
# print("Its status after prediction:")
# print(payroll_processed_df[payroll_processed_df['EmployeeID'] == anomaly_indices+1][['EmployeeID', 'anomaly_score', 'is_anomaly_predicted']])
```
The crucial part was then reviewing what the model flagged. I compared the `EmployeeID`s of the detected anomalies back to my original `payroll_df` to see their actual values. It was quite satisfying to see it pick up on some of the obvious errors I had planted, like an employee with a `BasePayRate` of $750/hour or someone with 120 `HoursWorked` in a week.

For example, one of my simulated anomalies was an employee whose `HoursWorked` was set to an extremely high value. The model successfully flagged this record. Another was a `BasePayRate` that was 10 times higher than typical for that department. The model caught that too.

There were, of course, some records flagged that weren't part of my *intentionally* created anomalies, and some of my planted anomalies were missed, especially the more subtle ones. This highlighted the challenge: the model finds things that are *mathematically* unusual based on the patterns it learned. Sometimes these align perfectly with what we consider errors, and sometimes they're just rare but valid data points, or the model's threshold isn't perfectly set.

**Challenges and What I Learned**

This project really hammered home that the definition of an "anomaly" can be fuzzy. What I simulated as an anomaly was based on my own rules, but in a real-world scenario, this would require significant domain expertise.

*   **The `contamination` parameter:** This was the most fiddly part. It directly influences sensitivity. Too high, and you get a flood of false positives. Too low, and you miss things. Iterative testing and looking at the anomaly scores rather than just the binary prediction helped.
*   **Feature Engineering:** The choice of features is critical. If I hadn't included `GrossPay` or `NetPay`, some anomalies related to their miscalculation might have been harder to detect. I considered creating more complex features, like comparing an individual's pay to the average for their department and state, but decided to keep it simpler for this iteration.
*   **Interpreting Anomalies:** Just getting a list of anomalous IDs isn't enough. The next step (which I only touched upon) is figuring out *why* they are anomalous. This usually meant going back to the raw data for those records and trying to spot what was off.
*   **One-Hot Encoding Explosion:** For features with many unique categories (like 'State', if I had used all 50 US states), one-hot encoding can lead to a lot of new features. This can sometimes make models slower or harder to interpret. For this project with only 10 states, it was manageable.

**Future Directions**

This was a great learning exercise in applying anomaly detection. If I were to take this further:
1.  **More Sophisticated Anomaly Simulation:** Create a wider variety of more subtle anomalies in the training data.
2.  **Feedback Loop:** In a real system, if an analyst confirms or rejects a flagged anomaly, that feedback could be used to retrain and improve the model over time.
3.  **Explore Other Models:** Try out other algorithms like One-Class SVM or even autoencoders (a type of neural network) to see how they compare.
4.  **Visualizations:** Developing some dashboards (perhaps with something like Matplotlib, Seaborn, or even a simple Flask app with Plotly) to visualize the detected anomalies and their features could make the results much more interpretable.
5.  **Explainability:** Delve into SHAP values or similar techniques to better understand *which features* are contributing most to a record being flagged as an anomaly.

Overall, building this anomalous payroll detector was a challenging but rewarding experience. It gave me practical insight into the workflow of an anomaly detection project, from data generation and preprocessing to model training and interpretation. It's clear that these tools aren't magic black boxes; they require careful setup and ongoing evaluation to be truly useful.