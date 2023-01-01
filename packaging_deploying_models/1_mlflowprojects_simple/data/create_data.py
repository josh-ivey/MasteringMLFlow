import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from scikit-learn
iris_data = load_iris()

# Convert the data to a Pandas dataframe
df = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
df['target'] = iris_data['target']

# Save the data to a CSV file
df.to_csv('iris.csv', index=False)
