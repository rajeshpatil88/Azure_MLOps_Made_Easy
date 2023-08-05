# train_model.py
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'simple_model.pkl')
