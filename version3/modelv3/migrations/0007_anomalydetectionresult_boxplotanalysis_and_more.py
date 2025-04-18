# Generated by Django 5.1.7 on 2025-03-29 01:28

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("modelv3", "0006_meanmedianmoderesults"),
    ]

    operations = [
        migrations.CreateModel(
            name="AnomalyDetectionResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("anomaly_scores", models.JSONField()),
                ("threshold", models.FloatField()),
                ("n_anomalies", models.IntegerField()),
                ("anomaly_ratio", models.FloatField()),
                ("feature_importance", models.JSONField()),
                ("predictions", models.JSONField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="BoxPlotAnalysis",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("feature_name", models.CharField(max_length=100)),
                ("boxplot_data", models.JSONField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="GradientBoostResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("accuracy", models.FloatField(default=0.0)),
                ("precision", models.FloatField(default=0.0)),
                ("recall", models.FloatField(default=0.0)),
                ("f1_score", models.FloatField(default=0.0)),
                ("support", models.FloatField(default=0.0)),
                ("macro_avg", models.FloatField(default=0.0)),
                ("weighted_avg", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="LinearRegressionResults",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("r2_score", models.FloatField(default=0.0)),
                ("mse", models.FloatField(default=0.0)),
                ("mae", models.FloatField(default=0.0)),
                ("root_mean_squared_error", models.FloatField(default=0.0)),
                ("adjusted_r2_score", models.FloatField(default=0.0)),
            ],
        ),
        migrations.CreateModel(
            name="OutlierDetectionResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("n_outliers", models.IntegerField()),
                ("outlier_ratio", models.FloatField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="StandardDeviationVarianceResults",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("standard_deviation", models.FloatField(default=0.0)),
                ("variance", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="FeatureImportance",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("feature_name", models.CharField(max_length=100)),
                ("importance", models.FloatField()),
                (
                    "gradient_boost_result",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="feature_importances",
                        to="modelv3.gradientboostresult",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="OutlierFeatureResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("feature_name", models.CharField(max_length=100)),
                ("lower_bound", models.FloatField()),
                ("upper_bound", models.FloatField()),
                ("n_outliers", models.IntegerField()),
                ("outlier_indices", models.JSONField()),
                (
                    "outlier_detection_result",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="feature_results",
                        to="modelv3.outlierdetectionresult",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Prediction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("predicted_value", models.FloatField()),
                ("actual_value", models.FloatField()),
                ("index", models.IntegerField()),
                (
                    "gradient_boost_result",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="predictions",
                        to="modelv3.gradientboostresult",
                    ),
                ),
            ],
        ),
    ]
