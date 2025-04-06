'''#task1 the impact of outliers on mean median mode..!!

#step1: create a database with outliers..!!

import statistics as stats

# Dataset without outliers
data = [10, 12, 11, 13, 12, 10, 14, 13, 12, 11]

# Add outliers
data_with_outliers = data + [100, 105]

print("Original Data:", data)
print("Data with Outliers:", data_with_outliers)

#step2: calculate mean median mode (before and after outlier)

def print_central_tendencies(dataset, label=""):
    print(f"\n{label} Central Tendency Measures:")
    print("Mean:", stats.mean(dataset))
    print("Median:", stats.median(dataset))
    print("Mode:", stats.mode(dataset))

# Without Outliers
print_central_tendencies(data, "Without Outliers")

# With Outliers
print_central_tendencies(data_with_outliers, "With Outliers")

#task2: measures of central tendency - mean median mole

#step1: create a dataset of test scores

import statistics as stats
import matplotlib.pyplot as plt

# Only use seaborn if it's installed; let's stick to matplotlib for simplicity
#import seaborn as sns

# Sample test scores
scores = [72, 85, 90, 88, 76, 95, 89, 78, 85, 91, 88, 85]

# Calculate statistics
mean_score = stats.mean(scores)
median_score = stats.median(scores)
mode_score = stats.mode(scores)

print("Mean:", mean_score)
print("Median:", median_score)
print("Mode:", mode_score)

# Plot histogram using matplotlib only
plt.hist(scores, bins=10, color='lightblue', edgecolor='black')

# Add lines for central tendencies
plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score}')
plt.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score}')
plt.axvline(mode_score, color='blue', linestyle='--', label=f'Mode: {mode_score}')

plt.title("Test Scores Distribution")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
#task 4
import scipy.stats as stats

# Sample scores before and after an intervention (e.g., training, treatment, etc.)
before_scores = [70, 75, 80, 85, 78, 74, 69, 90, 88, 76]
after_scores  = [74, 78, 85, 87, 82, 79, 72, 93, 90, 80]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(after_scores, before_scores)

# Output results
print("Paired t-test results")
print("---------------------")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("Result: Significant difference (reject null hypothesis)")
else:
    print("Result: No significant difference (fail to reject null hypothesis)")
import matplotlib.pyplot as plt

plt.plot(before_scores, label='Before', marker='o')
plt.plot(after_scores, label='After', marker='o')
plt.title("Before vs After Scores")
plt.xlabel("Student Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#task 5

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Sample data: Hours studied vs Test scores
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_scores   = [50, 55, 53, 60, 62, 65, 70, 72, 75, 78]

# Calculate correlation coefficient
correlation, _ = stats.pearsonr(hours_studied, test_scores)
print(f"Correlation (r): {correlation:.3f}")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(hours_studied, test_scores)
print(f"Regression equation: Score = {slope:.2f} * Hours + {intercept:.2f}")
print(f"R-squared: {r_value**2:.3f}")

# Predict scores
predicted_scores = [slope * x + intercept for x in hours_studied]

# Plot data and regression line
plt.scatter(hours_studied, test_scores, color='blue', label='Actual Scores')
plt.plot(hours_studied, predicted_scores, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Linear Regression: Hours Studied vs Test Scores')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#task6

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Assume mean and standard deviation (based on previous test scores)
mean = 80
std_dev = 5

# Calculate probability of scoring more than 85
prob = 1 - norm.cdf(85, loc=mean, scale=std_dev)
print(f"Probability of scoring more than 85: {prob:.4f}")

# Plot the normal distribution
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = norm.pdf(x, loc=mean, scale=std_dev)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Normal Distribution', color='skyblue')

# Shade area for P(X > 85)
x_fill = np.linspace(85, mean + 4*std_dev, 500)
y_fill = norm.pdf(x_fill, loc=mean, scale=std_dev)
plt.fill_between(x_fill, y_fill, color='red', alpha=0.5, label='P(X > 85)')

plt.title('Normal Distribution of Test Scores')
plt.xlabel('Test Score')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#task7

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given values
score = 88
mean = 80
std_dev = 5

# Calculate z-score
z_score = (score - mean) / std_dev
print(f"Z-score for {score}: {z_score:.2f}")

# Probability of scoring less than 88
prob = norm.cdf(z_score)
print(f"Probability of scoring less than {score}: {prob:.4f}")

# Plot normal distribution
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = norm.pdf(x, mean, std_dev)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Normal Distribution", color='skyblue')

# Mark and shade the area left of the z-score
x_fill = np.linspace(mean - 4*std_dev, score, 500)
y_fill = norm.pdf(x_fill, mean, std_dev)
plt.fill_between(x_fill, y_fill, color='green', alpha=0.5, label=f'P(X < {score})')

# Draw a vertical line at the z-score position
plt.axvline(score, color='red', linestyle='--', label=f'Score = {score}')

plt.title("Z-score on Normal Distribution")
plt.xlabel("Test Score")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#task8

import numpy as np
import matplotlib.pyplot as plt

# Sample test scores (include some possible outliers)
scores = [72, 85, 90, 88, 76, 95, 89, 78, 85, 91, 88, 85, 45, 99]

# Calculate Q1, Q3, and IQR
q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)
iqr = q3 - q1

# Define bounds for outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify outliers
outliers = [score for score in scores if score < lower_bound or score > upper_bound]

# Output results
print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Outliers: {outliers}")
