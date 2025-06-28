import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("Loading data...")
df = pd.read_csv("myo_gesture_data.csv")

X = df.drop("label", axis=1)  # EMG data
y = df["label"]  # Labels

# Split data into training and testing sets (optional, but good practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training KNeighborsClassifier...")
# You can adjust n_neighbors based on your data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save the trained model
model_filename = "gesture_classifier.pkl"
joblib.dump(knn, model_filename)
print(f"Model trained and saved as {model_filename}")
