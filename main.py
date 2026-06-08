import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sales_data.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove Duplicates
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# Save Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# -----------------------------
# VISUALIZATION 1
# -----------------------------

numeric_cols = df.select_dtypes(include=['number']).columns

if len(numeric_cols) > 0:

    plt.figure(figsize=(8,5))

    sns.histplot(df[numeric_cols[0]], kde=True)

    plt.title(f"Distribution of {numeric_cols[0]}")

    plt.savefig("distribution.png")

    plt.show()

# -----------------------------
# VISUALIZATION 2
# -----------------------------

if len(numeric_cols) > 1:
    plt.figure(figsize=(8,5))
    sns.scatterplot(
        x=df[numeric_cols[0]],
        y=df[numeric_cols[1]]
    )

    plt.title(
        f"{numeric_cols[0]} vs {numeric_cols[1]}"
    )

    plt.savefig("scatter_plot.png")

    plt.show()

# -----------------------------
# VISUALIZATION 3
# -----------------------------

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()
print("\nProject Completed Successfully!")