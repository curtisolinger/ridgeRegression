import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


rng = np.random.default_rng()
rng.integers(low=1, high=20, size=4)

data = {'11am': np.array(rng.integers(low=1, high=20, size=4)),
        '12pm': np.array(rng.integers(low=1, high=20, size=4)),
        '1pm': np.array(rng.integers(low=1, high=20, size=4)),
        '2pm': np.array(rng.integers(low=1, high=20, size=4)),
        '3pm': np.array(rng.integers(low=1, high=20, size=4)),
        'flash_demand': [200, 250, 300, 120]
}


X = np.array(data['3pm']).reshape(4,1)
y = np.array(data['flash_demand'])

reg = LinearRegression().fit(X, y)
 
# df = pd.DataFrame(data)
# df.plot.scatter(x='3pm', y='flash_demand')
 
# Generate predicted values for plotting the best-fit line
x_fit = np.linspace(np.array(data['3pm']).min(), np.array(data['3pm']).max(), 100).reshape(-1, 1) # Generate evenly spaced x values
y_fit = reg.predict(x_fit) # Compute corresponding y values

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(data['3pm'], data['flash_demand'])
ax.plot(x_fit, y_fit, color='red', label='Least Squares Fit') # Add regression line
ax.set_xlabel('Demand at 3pm')
ax.set_ylabel('flash_demand')
ax.set_title('Plot of demand at 3pm vs EOD flash_demand')
ax.legend()
plt.show()






# print(reg.coef_)

# print(X)

# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3

# print(X.shape)


observation_1  = np.array([10, 20, 22, 5])
observation_2 = np.array([6, 7, 8, 2])
observation_3 = np.array([11, 5, 1, 10])

df = pd.DataFrame([
    observation_1,
    observation_2], 
    columns=['A', 'B', 'C', 'D']
)

# print(df)

