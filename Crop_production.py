import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (IMPORTANT: use relative path for GitHub)
data = pd.read_csv(r"C:\Users\amarj\git-demo\upskillCampus\fixed_crop_production1.csv")

# Preview data
print(data.head())

# Features and target
X = data[['Area', 'Rainfall', 'Temperature']]
y = data['Production']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)

# ---------------- VISUALIZATIONS ---------------- #

# 1. Rainfall vs Production
plt.scatter(data['Rainfall'], data['Production'])
plt.xlabel("Rainfall")
plt.ylabel("Production")
plt.title("Rainfall vs Production")
plt.show()

# 2. Area vs Production
plt.scatter(data['Area'], data['Production'])
plt.xlabel("Area")
plt.ylabel("Production")
plt.title("Area vs Production")
plt.show()

# 3. Temperature vs Production
plt.scatter(data['Temperature'], data['Production'])
plt.xlabel("Temperature")
plt.ylabel("Production")
plt.title("Temperature vs Production")
plt.show()

# 4. Crop-wise Production
data.groupby('Crop')['Production'].mean().plot(kind='bar')
plt.title("Average Production by Crop")
plt.ylabel("Production")
plt.show()

# 5. State-wise Production
data.groupby('State')['Production'].sum().plot(kind='bar')
plt.title("Total Production by State")
plt.ylabel("Production")
plt.xticks(rotation=45)
plt.show()

# 6. Correlation Heatmap (IMPORTANT 🔥)
sns.heatmap(data[['Area','Rainfall','Temperature','Production']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()