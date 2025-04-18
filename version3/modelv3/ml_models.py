import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, confusion_matrix
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import django
import streamlit as st
import xgboost as xgb

st.set_page_config(
    layout="wide",
)



# project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'version3.settings')
django.setup()

from modelv3.api_key import OPENAI_API_KEY
from modelv3.models import PCAResult, AnovaResult, RandomForestResult, MeanMedianModeResults, StandardDeviationVarianceResults, LinearRegressionResults, AnomalyDetectionResult, GradientBoostResult, FeatureImportance, Prediction, OutlierDetectionResult, OutlierFeatureResult, BoxPlotAnalysis

client = OpenAI(api_key=OPENAI_API_KEY)


df = pd.read_csv('/Users/dog/Documents/Personal_Projects/Nikhil/version3/modelv3/predictive_maintenance.csv')
df.dropna(inplace=True)
numeric_features = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

st.write("# Automated Data Analysis")
st.write("## This AI automated data analysis model will help you analyze your data, answer your queries and help you understand your data better.")
uploaded_file = st.file_uploader('Upload your file here')
st.subheader("Dataset")
with st.expander("View the dataset here:"):
    st.write(df)

st.subheader("Filter data")
with st.expander("Filter your data over here:"):
    selected_features = st.multiselect("Select features to display", df.columns)
    filtered_df = df[selected_features]
    st.write(filtered_df)


# Load the dataset again after reset
file_path = "/Users/dog/Documents/Personal_Projects/Nikhil/version3/modelv3/predictive_maintenance.csv"
df = pd.read_csv(file_path)


# TODO 1: create functions for each analysis & store them in the django model.


    # TODO 7: Time Series Analysis
    # TODO 8: K-means Clustering
    # TODO 9: Gradient Boosting (XGBoost, LightGBM)
    # TODO 10: Autoencoders for anomaly detectio
    # TODO 11: LSTM for time series forecasting

def standard_deviation_variance():
    """
    This function runs standard deviation and variance analysis on the dataset.
    """
    # Standard deviation analysis
    std_deviation = df[numeric_features].std()
    variance = df[numeric_features].var()
    StandardDeviationVarianceResults.objects.create(
        standard_deviation=std_deviation,
        variance=variance
    )


def linear_regression():
    """
    This function runs linear regression analysis on the dataset.
    """
    # Linear regression analysis
    X = df[numeric_features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(mse)
    adjusted_r2_score = 1 - (1-r2_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    coefficients = pd.DataFrame({
        'Intercept': model.intercept_,
        'Feature': X.columns,
        'Coefficient': model.coef_
    })

    LinearRegressionResults.objects.create(
        r2_score=r2_score,
        mse=mse,
        mae=mae,
        root_mean_squared_error=root_mean_squared_error,
        adjusted_r2_score=adjusted_r2_score,
        coefficients=coefficients
    )

# TODO 5: Histogram
def Histogram():
    return None

def AnomalyDetection():
    """
    This function performs anomaly detection using Isolation Forest and stores the results.
    """
    # Prepare data
    X = df[numeric_features]
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X)
    
    # Calculate anomaly scores
    anomaly_scores = iso_forest.score_samples(X)
    
    # Calculate threshold (using 95th percentile)
    threshold = np.percentile(anomaly_scores, 5)
    
    # Identify anomalies
    predictions = iso_forest.predict(X)
    anomalies = X[predictions == -1]
    normal = X[predictions == 1]
    
    # Calculate metrics
    n_samples = len(X)
    n_anomalies = len(anomalies)
    anomaly_ratio = n_anomalies / n_samples
    
    # Calculate feature importance for anomalies
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': np.abs(iso_forest.feature_importances_)
    })
    
    # Store results in database
    AnomalyDetectionResult.objects.create(
        anomaly_scores=anomaly_scores.tolist(),
        threshold=threshold,
        n_anomalies=n_anomalies,
        anomaly_ratio=anomaly_ratio,
        feature_importance=feature_importance.to_dict(),
        predictions=predictions.tolist()
    )
    
def TimeSeriesAnalysis():
    return None

def GradientBoost():
    """
    This function performs gradient boosting analysis using XGBoost and stores the results.
    """
    # Prepare data
    X = df[numeric_features]
    y = df['Target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    macro_avg = precision_recall_fscore_support(y_test, y_pred, average='macro')
    weighted_avg = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create main result record
    result = GradientBoostResult.objects.create(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        support=support,
        macro_avg=macro_avg[0],
        weighted_avg=weighted_avg[0]
    )
    
    # Store feature importance
    for _, row in feature_importance.iterrows():
        FeatureImportance.objects.create(
            gradient_boost_result=result,
            feature_name=row['feature'],
            importance=row['importance']
        )
    
    # Store predictions
    for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
        Prediction.objects.create(
            gradient_boost_result=result,
            predicted_value=pred,
            actual_value=actual,
            index=i
        )
    

def Autoencoders():
    return None

def LSTM():
    return None

def mean_median_mode():
    """
    This function runs mean, median, and mode analysis on the dataset.
    """
    for feature in numeric_features:
        # Calculate statistics for each feature
        mean_value = df[feature].mean()
        median_value = df[feature].median()
        mode_value = df[feature].mode().iloc[0]  # Get first mode if multiple exist

        # Store results in database
        MeanMedianModeResults.objects.create(
            mean_values=mean_value,
            median_values=median_value,
            mode_values=mode_value
        )


def box_plot_analysis():
    """
    This function runs box plot analysis on the dataset and stores the results.
    """
    for feature in numeric_features:
        # Calculate box plot statistics
        boxplot_data = {
            'min': float(df[feature].min()),
            'q1': float(df[feature].quantile(0.25)),
            'median': float(df[feature].median()),
            'q3': float(df[feature].quantile(0.75)),
            'max': float(df[feature].max()),
            'iqr': float(df[feature].quantile(0.75) - df[feature].quantile(0.25)),
            'lower_whisker': float(df[feature].quantile(0.25) - 1.5 * (df[feature].quantile(0.75) - df[feature].quantile(0.25))),
            'upper_whisker': float(df[feature].quantile(0.75) + 1.5 * (df[feature].quantile(0.75) - df[feature].quantile(0.25))),
            'outliers': [float(x) for x in df[feature][(df[feature] < df[feature].quantile(0.25) - 1.5 * (df[feature].quantile(0.75) - df[feature].quantile(0.25))) | 
                                (df[feature] > df[feature].quantile(0.75) + 1.5 * (df[feature].quantile(0.75) - df[feature].quantile(0.25)))].tolist()]
        }
        
        # Store results in database
        BoxPlotAnalysis.objects.create(
            feature_name=feature,
            boxplot_data=boxplot_data
        )
        

# There is no time column in the dataset, so we cannot run moving average analysis.

def MovingAverage():
    """
    This function runs moving average analysis on the dataset.
    """
    # Moving average analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in numeric_features:
        df[f'{feature}_moving_avg'] = df[feature].rolling(window=5).mean()
        sns.lineplot(x='Time', y=f'{feature}_moving_avg', data=df, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Moving Average')
    ax.set_title('Moving Average of Features')
    st.pyplot(fig)

# There is no time column in the dataset, so we cannot run Z-score analysisd.
def ZScore():
    """
    This function runs Z-score analysis on the dataset.
    """
    # Z-score analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in numeric_features:
        df[f'{feature}_zscore'] = (df[feature] - df[feature].mean()) / df[feature].std()
        sns.lineplot(x='Time', y=f'{feature}_zscore', data=df, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Z-score')
    ax.set_title('Z-score of Features')
    st.pyplot(fig)

def CorrelationAnalysis():
    """
    This function runs correlation analysis on the dataset.
    """
    # Correlation analysis
    fig, ax = plt.subplots(figsize=(10, 6)) 
    sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Features')
    st.pyplot(fig)    

def OutlierDetection():
    """
    This function performs outlier detection using IQR method and stores the results.
    """
    # Create main result record
    result = OutlierDetectionResult.objects.create(
        n_outliers=0,  # Will be updated after processing
        outlier_ratio=0.0  # Will be updated after processing
    )
    
    total_outliers = 0
    total_samples = len(df)
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Process each feature
    for feature in numeric_features:
        # Calculate quartiles and IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_indices = outliers.index.tolist()
        n_outliers = len(outliers)
        total_outliers += n_outliers
        
        # Store feature-specific results
        OutlierFeatureResult.objects.create(
            outlier_detection_result=result,
            feature_name=feature,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_outliers=n_outliers,
            outlier_indices=outlier_indices
        )
        
        # Add to box plot
        sns.boxplot(y=df[feature], ax=ax)
        ax.set_title(f'Outlier Detection for {feature}')
    
    # Update main result with total counts
    result.n_outliers = total_outliers
    result.outlier_ratio = total_outliers / total_samples
    result.save()
    
    


def PCAA():
    # Standard scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])

    # First run PCA with all components
    pca_all = PCA().fit(scaled_data)

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)

    # Pick the correct number of components that explain 95% of variance
    n_components = np.argmax(cumulative_variance >= 0.95) + 1

    # Now run PCA with optimal number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

    # Store PCA results in database
    for _, row in reduced_df.iterrows():
        PCAResult.objects.create(
            pc1=row['PC1'],
            pc2=row['PC2']
        )

    # Get PCA results from database for LLM analysis
    pca_results = PCAResult.objects.all()[:200] # all()[:150] for first 150 results
    pca_data = "\n".join([f"Entry {i}: PC1={result.pc1:.4f}, PC2={result.pc2:.4f}"
                        for i, result in enumerate(pca_results, 1)])    

def AnovaAnalysis():
    # ANOVA Analysis 

    anova_results = []
    for feature in numeric_features:
        groups = [group[feature].values for name, group in df.groupby('Target')]
        f_statistic, p_value = stats.f_oneway(*groups)
        significance = p_value < 0.05
        
        # Store ANOVA result in database
        AnovaResult.objects.create(
            feature_name=feature,
            f_statistic=f_statistic,
            p_value=p_value,
            significance=significance
        )
        
        anova_results.append({
            'feature': feature,
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significance': significance
        })

    # Get ANOVA results from database for LLM analysis
    anova_data = AnovaResult.objects.all()
    anova_data_str = "\n".join([
        f"Feature: {result.feature_name}\n"
        f"F-statistic: {result.f_statistic:.4f}\n"
        f"p-value: {result.p_value:.4f}\n"
        f"Significant: {'Yes' if result.significance else 'No'}\n"
        for result in anova_data
    ])

def RandomForestModel():    # Random Forest Model
    X = df[numeric_features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    macro_avg = precision_recall_fscore_support(y_test, y_pred, average='macro')
    weighted_avg = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Store results in database
    RandomForestResult.objects.create(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        support=support,
        macro_avg=macro_avg[0],  # Using precision from macro average
        weighted_avg=weighted_avg[0],  # Using precision from weighted average
        predicted_value=y_pred[0],  # Store first prediction
        actual_value=y_test.iloc[0]  # Store first actual value
    )
    return X_test, y_pred


def display_graph():
    # Get ANOVA results from database for plotting
    anova_data = AnovaResult.objects.all()
    anova_plot_data = pd.DataFrame([
        {
            'feature': result.feature_name,
            'p_value': result.p_value
        }
        for result in anova_data
    ])
    
    #Create bar plot of ANOVA results
    fig_anova, ax_anova = plt.subplots(figsize=(10, 6))
    anova_plot_data['-log10(p_value)'] = -np.log10(anova_plot_data['p_value'])
    ax_anova.bar(anova_plot_data['feature'], anova_plot_data['-log10(p_value)'])
    ax_anova.set_xlabel('Features')
    ax_anova.set_ylabel('-log10(p-value)')
    ax_anova.set_title('ANOVA Results: Feature Significance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_anova)     
    st.bar_chart(anova_plot_data, x='feature', y= '-log10(p_value)', color='feature')
    
    # # Create scatter plot of PCA results
    # pca_results = PCAResult.objects.all()[:500]
    # pca_df = pd.DataFrame([(result.pc1, result.pc2) for result in pca_results], columns=['PC1', 'PC2'])
    # # fig, ax = plt.subplots()
    # # ax.scatter(pca_df['PC1'], pca_df['PC2'])
    # # ax.set_xlabel('Principal Component 1')
    # # ax.set_ylabel('Principal Component 2') 
    # # ax.set_title('PCA Results Visualization')
    # # st.pyplot(fig)
    # st.scatter_chart(pca_df, x='PC1', y='PC2')

    # #scatter plot of Random Forest Model results plot from django model
    # random_forest_results = RandomForestResult.objects.all()
    # rf_df = pd.DataFrame(list(random_forest_results.values()))
    # st.scatter_chart(rf_df, x='predicted_value', y='actual_value')


def run_simple_analysis():
    """
    This function will run all the simple analysis functions(box plot analysis,
      moving average analysis, Z-score analysis, correlation analysis, outlier detection, 
      historgram analysis)
    """
    mean_median_mode()
    box_plot_analysis()
    OutlierDetection()
    PCAA()
    AnovaAnalysis()
    LinearRegression()

    

def llm_analysis(ps_prompt):
    anova_data = AnovaResult.objects.all()
    pca_data = PCAResult.objects.all()
    linear_regression_data = LinearRegressionResults.objects.all()
    outlier_detection_data = OutlierDetectionResult.objects.all()
    mean_median_mode_data = MeanMedianModeResults.objects.all()
    box_plot_data = BoxPlotAnalysis.objects.all()
    

    # random_forest_data = RandomForestResult.objects.all()
    prompt = f"""
    You are an expert data analyst, and you must analyze and respond to the following concisely and under 100 words and in bullet points:
    Go through the below analysis data and the problem statement,

    {anova_data}
    {pca_data}
    {linear_regression_data}
    {outlier_detection_data}
    {mean_median_mode_data}
    {box_plot_data}
    
    problem statement: {ps_prompt}
    Question for you: Is the analysis data provided to you is enough to answer the problem statement?.
    If not, what additional data do you need?. And from just the data provided, answer the problem statement.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500
    )
    return response.choices[0].message.content


    """
    # TODO :is the given data analysis answering the problem statement.
    # TODO : 

    You are an expert data analyst, and you must analyze and respond to the following 3 points and the problem statement in concisely and under 250 words:
    1. Answer the problem statement: "{ps_prompt}", Key patterns in the data and any data clusters.
    2. Which features are most significant and how did you come to this conclusion?
    3. How each feature contributes to the problem & Practical implications for the problem.
    """




'''
Prompt:

Which operational factors (temperature, speed, torque, etc..) contribute the most to machine failures. Do some root cause analysis

'''
problem_stmnt = st.text_area("Enter your problem statement here:")
start_analysis = st.button("Start Analysis")
if problem_stmnt and start_analysis:
    col1, col2 = st.columns(2)
    with st.spinner("Analyzing..."):
        with col1:
            run_simple_analysis()
            llm_response = llm_analysis(problem_stmnt)
            st.write(llm_response)
        with col2:
            display_graph()
       



