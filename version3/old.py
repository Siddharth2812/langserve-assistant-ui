
# Get problem statement = 

#
# Which operational factors (temperature, speed, torque, etc..) contribute the most to machine failures

ps_prompt = input("Provide the problem statement for the dataset: ")


# Ask user if they want to choose analysis or let LLM decide
choice_prompt = """
Would you like to:
1. Choose the analysis yourself
2. Let the AI recommend the most appropriate analysis

Enter 1 or 2: """
user_choice = input(choice_prompt).strip()


# Load data
df = pd.read_csv('/Users/nikhiltirunagiri/Documents/project1/version3/modelv3/predictive_maintenance.csv')
df.dropna(inplace=True)

numeric_features = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

# Determine which analysis to perform
if user_choice == "1":
    # User chooses analysis
    analysis_prompt = """
    Choose the analysis to perform:
    1. PCA (Principal Component Analysis) - for dimensionality reduction and pattern identification
    2. ANOVA (Analysis of Variance) - for feature importance and statistical significance
    3. Random Forest - for classification

    Enter 1 or 2 or 3: """
    analysis_choice = input(analysis_prompt).strip()
else:
    # Let LLM choose analysis
    analysis_prompt = f"""
    Based on this problem statement: "{ps_prompt}"

    Which type of analysis would be most appropriate? Choose one:
    1. PCA (Principal Component Analysis) - for dimensionality reduction and pattern identification
    2. ANOVA (Analysis of Variance) - for feature importance and statistical significance
    3. Random Forest - for classification

    Respond with just the number (1 or 2 or 3).
    """
    analysis_choice = llm.predict(analysis_prompt).strip()
    print(f"\nAI recommended analysis: {'PCA' if analysis_choice == '1' else'ANOVA' if analysis_choice == '2' else 'Random Forest'}")

# Perform chosen analysis
if analysis_choice == "1":
    # PCA Analysis
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    
    # Store PCA results in database
    for _, row in reduced_df.iterrows():
        PCAResult.objects.create(
            pc1=row['PC1'],
            pc2=row['PC2']
        )
    
    # Get PCA results from database for LLM analysis
    pca_results = PCAResult.objects.all()[:150]  # Get first 150 results
    pca_data = "\n".join([f"Entry {i}: PC1={result.pc1:.4f}, PC2={result.pc2:.4f}" 
                         for i, result in enumerate(pca_results, 1)])
    
    # Get LLM insights
    insight_prompt = f"""
    Based on this problem statement: "{ps_prompt}"
    
    Here are the PCA results from the database:
    {pca_data}
    
    Please analyze:
    1. Key patterns in the data
    2. How the data clusters
    3. Practical implications for the problem
    """

elif analysis_choice == "2":
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
    
    # Get LLM insights
    insight_prompt = f"""
    Based on this problem statement: "{ps_prompt}"
    
    Here are the ANOVA results from the database:
    {anova_data_str}
    
    Please analyze:
    1. Which features are most significant
    2. How each feature contributes to the problem
    3. Practical implications for the problem
    """
elif analysis_choice == "3":
    # Random Forest Analysis
    # Split data into training and testing sets
    X = df[numeric_features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Save model
    with open("random_forest_model.txt", "w") as file:
        file.write(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Get LLM insights
    insight_prompt = f"""
    Based on this problem statement: "{ps_prompt}"  
    
    Here are the Random Forest results:
    {y_pred}
    
    Please analyze:
    1. Model performance
    2. Feature importance
    3. Practical implications for the problem
    """

# Get and save LLM insights
response = llm.predict(insight_prompt)
with open("llm_analysis.txt", "w") as file:
    file.write(f"Analysis for: {ps_prompt}\n\n")
    file.write(response)







