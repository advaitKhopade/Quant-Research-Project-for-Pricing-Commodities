import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define the log_likelihood function as before.
def log_likelihood(n, k):
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)

# The calculate_breakpoints function remains the same.
def calculate_breakpoints(df, r):
    x = df['default'].tolist()
    y = df['fico_score'].tolist()
    n = len(x)
    
    default = [0 for _ in range(851)]
    total = [0 for _ in range(851)]
    
    for i in range(n):
        y[i] = int(y[i])
        default[y[i] - 300] += x[i]
        total[y[i] - 300] += 1
        
    for i in range(1, 551):  # Fixed the range to start from 1 instead of 0
        default[i] += default[i - 1]
        total[i] += total[i - 1]
    
    dp = [[[-10**18, 0] for _ in range(551)] for _ in range(r + 1)]
    
    for i in range(r + 1):
        for j in range(551):
            if i == 0:
                dp[i][j][0] = 0
            else:
                for k in range(j):
                    if total[j] == total[k]:
                        continue
                    if i == 1:
                        dp[i][j][0] = log_likelihood(total[j], default[j])
                    else:
                        if dp[i][j][0] < (dp[i - 1][k][0] + log_likelihood(total[j] - total[k], default[j] - default[k])):
                            dp[i][j][0] = dp[i - 1][k][0] + log_likelihood(total[j] - total[k], default[j] - default[k])
                            dp[i][j][1] = k
    return dp

# The get_buckets function also remains the same.
def get_buckets(dp, r):
    l = []
    k = 550
    while r > 0:  # Changed the condition to r > 0 to ensure the loop exits correctly.
        l.append(k + 300)
        k = dp[r][k][1]
        r -= 1
    l.reverse()  # Reverse the list to have it in ascending order.
    return l

# Assign_to_bucket function remains unchanged.
def assign_to_bucket(score, buckets):
    for i in range(len(buckets) - 1):
        if score >= buckets[i] and score < buckets[i + 1]:
            return i
    return len(buckets) - 1

# Ensure you have the 'Task 3 and 4_Loan_Data.csv' file in the correct folder.
def main():
    # Load the data.
    df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
    
    r = 10
    dp = calculate_breakpoints(df, r)
    breakpoints = get_buckets(dp, r)

    # Apply the function correctly.
    df['fico_score_bucket'] = df['fico_score'].apply(lambda x: assign_to_bucket(x, breakpoints))

    # Corrected feature selection logic.
    features = ['fico_score_bucket']  # Assuming you want to include only this feature for dummy encoding.
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['default']
    
    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10000)
    
    # Train the model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model. You need to replace these with your implementation or appropriate library calls.
    accuracy = model.score(X_test, y_test)  # Using model's score method for simplicity.
    print(f'Model Accuracy: {accuracy}')
    
    # Visualization.
    plt.figure(figsize=(10, 6))
    df['fico_score'].hist(bins=30, alpha=0.5)
    for breakpoint in breakpoints:
        plt.axvline(x=breakpoint, color='red', linestyle='--')
    plt.title('Distribution of FICO Scores with Bucket Thresholds')
    plt.xlabel('FICO Score')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
