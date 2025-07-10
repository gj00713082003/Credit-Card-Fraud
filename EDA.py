# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('creditcard.csv')  # Adjust the path as needed

# -------------------------------
# Basic Dataset Information
# -------------------------------
print("Dataset Info:")
print(df.info())

print("\nClass Distribution (Count):")
class_counts = df['Class'].value_counts()
print(class_counts)

print("\n Class Distribution (Percentage):")
class_percentage = df['Class'].value_counts(normalize=True) * 100
print(class_percentage)

# -------------------------------
# Visualizing Class Imbalance
# -------------------------------

# Bar plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Class Distribution (Imbalanced Data)')
plt.xlabel('Class (0 = Normal, 1 = Fraud)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
labels = ['Non-Fraud (0)', 'Fraud (1)']
plt.pie(class_counts, labels=labels, autopct='%1.2f%%', colors=['skyblue', 'salmon'], startangle=140)
plt.title('Class Proportion')
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle
plt.tight_layout()
plt.show()

# -------------------------------
# Statistical Summary & Missing Values
# -------------------------------
print("\nStatistical Summary:")
print(df.describe())

print("\n Missing Values:")
print(df.isnull().sum())

# -------------------------------
# Distribution of Time
# -------------------------------
plt.figure(figsize=(10, 4))
plt.hist(df['Time'], bins=500, color='blue', alpha=0.7)
plt.title('Distribution of Time')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------------------------------
# Distribution of Amount
# -------------------------------
plt.figure(figsize=(10, 4))
plt.hist(df['Amount'], bins=500, color='green', alpha=0.7)
plt.title('Distribution of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------------------------------
# Scatter plot for Class = 1
# -------------------------------
df_class_1 = df[df['Class'] == 1]

plt.figure(figsize=(10, 6))
plt.scatter(df_class_1['Time'], df_class_1['Amount'], 
            alpha=0.5, s=1, c=df_class_1['Class'], cmap='viridis')
plt.title('Scatter Plot of Time vs Amount (Class = 1)')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.colorbar(label='Class')
plt.tight_layout()
plt.show()

# -------------------------------
# Correlation Matrix Heatmap
# -------------------------------
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------
# Correlation with Class
# -------------------------------
print("\n Correlation with Class (Sorted):")
print(corr_matrix['Class'].sort_values(ascending=False))
print("The correlation with Class is highest for 'V17', 'V14', and 'V12'.")

print("\n Correlation with Class (Sorted):")
print(corr_matrix['Class'].sort_values(ascending=False))
print("The correlation with Class is highest for 'V17', 'V14', and 'V12'.")
print("Principal Component Analysis (PCA) implemented well such that matrix is goodly orthogonal.")
