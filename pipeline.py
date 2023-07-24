import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline


from sklearn.exceptions import ConvergenceWarning

df = pd.read_csv('german_credit_data.csv')

# Replace spaces with underscores in column names
df.columns = df.columns.str.replace(' ', '_')

print(df)

# Determining number of missing values for each column in our dataset.
missing_values = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

for column in df.columns:
    if df[column].isna().sum() != 0:
        missing = df[column].isna().sum()
        portion = (missing / df.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")


def visualize_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values.plot(kind='bar', figsize=(10, 6))
    plt.title('Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Missing Value Count')
    plt.xticks(rotation=45)
    plt.show()
visualize_missing_values(df)


def encode_categorical_columns(df):
    encoded_data = df.copy()

    for column in encoded_data.columns:
        if encoded_data[column].dtype == 'object':
            encoder = LabelEncoder()
            encoded_data[column] = encoder.fit_transform(encoded_data[column].astype(str))

    return encoded_data
encode_data = encode_categorical_columns(df)

df = encode_data






# this repeat fucntion will check if there is still missing data in the file
for column in df.columns:
    if df[column].isna().sum() != 0:
        missing = df[column].isna().sum()
        portion = (missing / df.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")



def test_target_variable_performance(data, pipeline):
    # Get the columns of the dataset
    columns = data.columns

    # Store results
    results = {}

    # Iterate through each column as the target variable
    for col in columns:
        # Split the data into features (X) and target variable (y)
        X = data.drop(columns=[col])
        y = data[col]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)

        # Initialize the model
        model_instance = pipeline

        # Train the model
        model_instance.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model_instance.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store the result
        results[col] = accuracy

    return results

# ... (previous code)

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define the pipeline steps (transformers only, no classifiers)
preprocessor = make_pipeline(
    SimpleImputer(strategy='mean'),  # Handle missing values (transformer)
    StandardScaler()  # Scale the features (transformer)
)

# List of classifiers to test
classifiers = [
    LogisticRegression(solver='saga', max_iter=1000),
    RandomForestClassifier(),
    GradientBoostingClassifier()
]

results = {}

# Iterate over classifiers
for classifier in classifiers:
    # Create a new pipeline for each classifier
    pipeline = make_pipeline(preprocessor, classifier)

    # Assuming 'data' is your pandas DataFrame and 'model' is your chosen ML model instance (e.g., LogisticRegression)
    classifier_name = classifier.__class__.__name__
    model_results = test_target_variable_performance(df, pipeline)
    results[classifier_name] = model_results

# Print the results for each classifier
for classifier, model_results in results.items():
    best_model_feature, best_accuracy = max(model_results.items(), key=lambda x: x[1])
    print(f"The best model for {classifier} with accuracy: {best_model_feature} - {best_accuracy}")
