import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Find the maximum number of features across all samples
max_num_features = max(len(sample) for sample in data)

# Initialize an array to store the data with consistent dimensions
data_array = np.zeros((len(data), max_num_features))

# Populate the data array by padding or truncating samples as needed
for i, sample in enumerate(data):
    data_array[i, :len(sample)] = sample[:max_num_features]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model
model = RandomForestClassifier()

# Fit the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
"""import pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict.keys())
print(data_dict['labels'])"""
