# Generated by Django 5.1.7 on 2025-03-19 01:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("modelv3", "0002_anovadata"),
    ]

    operations = [
        migrations.CreateModel(
            name="AnovaResult",
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
                ("f_statistic", models.FloatField()),
                ("p_value", models.FloatField()),
                ("significance", models.BooleanField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="PCAResult",
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
                ("pc1", models.FloatField()),
                ("pc2", models.FloatField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name="AnovaData",
        ),
        migrations.DeleteModel(
            name="PCAData",
        ),
    ]
