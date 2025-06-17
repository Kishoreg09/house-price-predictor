# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data: [area, bedrooms, bathrooms, parking]
X = np.array([
    [1000, 2, 1, 1],
    [1500, 3, 2, 2],
    [2000, 4, 3, 2],
    [800, 2, 1, 0],
    [1200, 3, 2, 1]
])

# Target price
y = np.array([10000000, 15000000, 20000000, 8500000, 13000000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model to a file
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)
