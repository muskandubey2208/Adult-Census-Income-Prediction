import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('adult.csv')

# Preprocessing
# Convert categorical variables to numeric
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Split the data into features and target
X = data.drop('salary', axis=1)
y = data['salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Use the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

high_income_indices = (predictions == '>50K')
high_income_data = X_test[high_income_indices]

# Make predictions
predictions = model.predict(X_test)
# Evaluate the model
print('Accuracy:', accuracy_score(y_test, predictions))
