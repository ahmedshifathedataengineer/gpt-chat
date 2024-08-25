import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore

# Step 2: Load the Dataset
df = pd.read_csv(r"C:\Users\hurri\OneDrive\Desktop\Titanic-Dataset.csv")

# Step 3.1: Understand the Data Structure
print(df.info())
print(df.describe())

# Step 3.2: Check for Missing Values
print(df.isnull().sum())

# Step 3.3.1: Survival Rate by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# Step 3.3.2: Survival Rate by Passenger Class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Step 3.3.3: Age Distribution and Survival
sns.histplot(df[df['Survived'] == 1]['Age'], kde=False, bins=30, label='Survived')
sns.histplot(df[df['Survived'] == 0]['Age'], kde=False, bins=30, label='Not Survived')
plt.legend()
plt.title('Age Distribution by Survival')
plt.show()

# Step 3.3.4: Survival Rate by Embarkation Point
sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title('Survival Rate by Embarkation Point')
plt.show()

# Step 4.1: Handle Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)

# Step 4.2: Encode Categorical Variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Step 4.3: Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 5: Building the Model
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predict and Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
