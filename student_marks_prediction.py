import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("===== Student Marks Prediction System =====")

# Step 1: Create dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 2.5, 4.5, 5.5, 6.5, 3.5, 7.5, 1.5],
    "Attendance":    [55, 60, 65, 70, 75, 80, 85, 90, 62, 72, 78, 82, 68, 88, 58],
    "Marks":         [40, 45, 50, 55, 60, 65, 70, 80, 48, 58, 63, 68, 53, 75, 42]
}

df = pd.DataFrame(data)

# Step 2: Show dataset
print("\nDataset:")
print(df)

# Step 3: Input and output
X = df[["Hours_Studied", "Attendance"]]
y = df["Marks"]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nModel Evaluation:")
print("Actual Marks   :", list(y_test))
y_pred = [float(i) for i in y_pred]
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Step 8: Predict new student marks
while True:
    choice = input("\nDo you want to predict marks for a student? (yes/no): ").lower()

    if choice == "yes":
        hours = float(input("Enter study hours: "))
        attendance = float(input("Enter attendance percentage: "))

        new_data = pd.DataFrame([[hours, attendance]], columns=["Hours_Studied", "Attendance"])
        predicted_marks = model.predict(new_data)

        print("Predicted Marks:", round(predicted_marks[0], 2))

    elif choice == "no":
        print("Exiting prediction system.")
        break

    else:
        print("Please enter yes or no.")

# Step 9: Save dataset to CSV
df.to_csv("student_data.csv", index=False)
print("\nDataset saved as student_data.csv")

# Step 10: Graph 1 - Hours Studied vs Marks
plt.figure(figsize=(6, 4))
plt.scatter(df["Hours_Studied"], df["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")
plt.grid(True)
plt.show()

# Step 11: Graph 2 - Attendance vs Marks
plt.figure(figsize=(6, 4))
plt.scatter(df["Attendance"], df["Marks"])
plt.xlabel("Attendance Percentage")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.grid(True)
plt.show()
