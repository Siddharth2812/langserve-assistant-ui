from django.db import models
import numpy as np
from django.db.models import JSONField

# class PCAData(models.Model):
#     # Metadata

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Output:
# reduced dataframe:
#         PC1       PC2
# 0 -1.093847 -0.854210
# 1 -1.602678 -0.215765
# 2 -1.599406 -0.401641
# 3 -1.230438 -0.664604
# 4 -1.289814 -0.535153

# store the above data in a model.


class PCAResult(models.Model):
    pc1 = models.FloatField()
    pc2 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PCA Result {self.id} (PC1={self.pc1:.4f}, PC2={self.pc2:.4f})"

class AnovaResult(models.Model):
    feature_name = models.CharField(max_length=100)
    f_statistic = models.FloatField()
    p_value = models.FloatField()
    significance = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ANOVA Result for {self.feature_name} (p={self.p_value:.4f})"

class RandomForestResult(models.Model):
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    support = models.FloatField(default=0.0)
    macro_avg = models.FloatField(default=0.0)
    weighted_avg = models.FloatField(default=0.0)
    predicted_value = models.FloatField(default=0.0)
    actual_value = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Random Forest Result (Accuracy={self.accuracy:.4f})"
    
class MeanMedianModeResults(models.Model):
    mean_values = models.FloatField(default=0.0)
    median_values = models.FloatField(default=0.0)
    mode_values = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Mean Median Mode Results (Mean={self.mean_values:.4f}, Median={self.median_values:.4f}, Mode={self.mode_values:.4f})"

class StandardDeviationVarianceResults(models.Model):
    standard_deviation = models.FloatField(default=0.0)
    variance = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Standard Deviation Variance Results (Standard Deviation={self.standard_deviation:.4f}, Variance={self.variance:.4f})"
        
class LinearRegressionResults(models.Model):
    r2_score = models.FloatField(default=0.0)
    mse = models.FloatField(default=0.0)
    mae = models.FloatField(default=0.0)
    root_mean_squared_error = models.FloatField(default=0.0)
    adjusted_r2_score = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"Linear Regression Results (R2 Score={self.r2_score:.4f}, MSE={self.mse:.4f}, MAE={self.mae:.4f}, Root Mean Squared Error={self.root_mean_squared_error:.4f}, Adjusted R2 Score={self.adjusted_r2_score:.4f})"

class AnomalyDetectionResult(models.Model):
    anomaly_scores = models.JSONField()  # List of anomaly scores
    threshold = models.FloatField()      # Threshold for anomaly detection
    n_anomalies = models.IntegerField()  # Number of anomalies detected
    anomaly_ratio = models.FloatField()  # Ratio of anomalies to total samples
    feature_importance = models.JSONField()  # Dictionary of feature importance
    predictions = models.JSONField()      # List of predictions (-1 for anomaly, 1 for normal)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Anomaly Detection Result - {self.created_at}"

class GradientBoostResult(models.Model):
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    support = models.FloatField(default=0.0)
    macro_avg = models.FloatField(default=0.0)
    weighted_avg = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Gradient Boost Result (Accuracy={self.accuracy:.4f})"

class FeatureImportance(models.Model):
    gradient_boost_result = models.ForeignKey(GradientBoostResult, on_delete=models.CASCADE, related_name='feature_importances')
    feature_name = models.CharField(max_length=100)
    importance = models.FloatField()

    def __str__(self):
        return f"{self.feature_name}: {self.importance:.4f}"

class Prediction(models.Model):
    gradient_boost_result = models.ForeignKey(GradientBoostResult, on_delete=models.CASCADE, related_name='predictions')
    predicted_value = models.FloatField()
    actual_value = models.FloatField()
    index = models.IntegerField()  # To maintain order of predictions

    def __str__(self):
        return f"Predicted: {self.predicted_value}, Actual: {self.actual_value}"

class OutlierDetectionResult(models.Model):
    n_outliers = models.IntegerField()  # Number of outliers detected
    outlier_ratio = models.FloatField()  # Ratio of outliers to total samples
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Outlier Detection Result - {self.created_at}"

class OutlierFeatureResult(models.Model):
    outlier_detection_result = models.ForeignKey(OutlierDetectionResult, on_delete=models.CASCADE, related_name='feature_results')
    feature_name = models.CharField(max_length=100)
    lower_bound = models.FloatField()  # Lower bound for outliers (Q1 - 1.5*IQR)
    upper_bound = models.FloatField()  # Upper bound for outliers (Q3 + 1.5*IQR)
    n_outliers = models.IntegerField()  # Number of outliers for this feature
    outlier_indices = models.JSONField()  # List of indices where outliers were found

    def __str__(self):
        return f"Outlier Result for {self.feature_name}"

class BoxPlotAnalysis(models.Model):
    feature_name = models.CharField(max_length=100)
    boxplot_data = models.JSONField()  # List of boxplot data points
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Box Plot Analysis for {self.feature_name}"
    



    