import numpy as np
import scipy.stats as stats

# 1. Generate sample data with outliers
np.random.seed(42)
data = np.random.normal(0, 1, 100)
outliers = np.array([4.5, 5.0, -4.0]) # Outliers
data = np.concatenate((data, outliers))

# 2. Define a reference distribution (normal distribution)
reference = np.random.normal(0, 1, 1000)

# 3. Perform the K-S test
# The statistic D_n is the max distance between CDFs
d_stat, p_value = stats.ks_2samp(data, reference)
print(f"K-S Statistic (Dn): {d_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# 4. Outlier Detection
# If p-value < 0.05, data differs significantly from reference
if p_value < 0.05:
    print("Significant differences found. Potential outliers exist.")
    # Simple approach: Identify data points outside the 95% confidence interval
    # of the reference distribution
    lower_bound = np.percentile(reference, 2.5)
    upper_bound = np.percentile(reference, 97.5)
    detected_outliers = data[(data < lower_bound) | (data > upper_bound)]
    print(f"Detected outliers based on threshold: {detected_outliers}")
else:
    print("No significant outliers detected.")
s