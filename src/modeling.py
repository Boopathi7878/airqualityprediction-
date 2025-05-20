import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"D:\\Codes\\Projects\\ML\\air_quality_model\data\\processed\\processed.csv")

# Check column names
print("Available columns:", df.columns.tolist())

# Define features and target based on actual data
numeric_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'CO']
target = 'AQI'

# Drop rows with missing values in relevant columns
df = df.dropna(subset=numeric_features + [target])

# Split data
X = df[numeric_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
pipeline = Pipeline(steps=[('preprocessing', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'models/random_forest_pipeline.pkl')

print("Model trained and saved to models/random_forest_pipeline.pkl")
