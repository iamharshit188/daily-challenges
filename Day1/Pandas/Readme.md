1. Introduction to Pandas

    Pandas is a Python library purpose-built for data manipulation and analysis.

    It supports loading, cleaning, transforming, visualizing, and exporting data—tasks required before any machine learning model is trained

    .

    The library's core data structures are:

        Series: 1-dimensional labeled array.

        DataFrame: 2-dimensional labeled table (like an Excel sheet, central in ML workflows).

python
import pandas as pd

2. Installing and Importing Pandas

python
# Command-line install (if needed)
!pip install pandas

# Import in your Python file or notebook
import pandas as pd

3. Series and DataFrame Basics
Series

    Purpose: Represents a single column of data, with meaningful labels ("index").

    ML Use: Store a single feature/label vector.

python
import pandas as pd
ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(ser)

DataFrame

    Purpose: Main data structure—a table of rows and columns.

    ML Use: Store your dataset (features & labels).

python
data = {
    "age": [23, 25, 31],
    "salary": [50000, 60000, 75000]
}
df = pd.DataFrame(data)
print(df)

4. Loading and Saving Data
Import Data

    CSV files (most ML datasets):

python
df = pd.read_csv('data.csv')

    Excel files:

python
df = pd.read_excel('data.xlsx')

    JSON files:

python
df = pd.read_json('data.json')

Export Data

python
df.to_csv('out.csv', index=False)
df.to_excel('out.xlsx', index=False)
df.to_json('out.json')

ML Application: Use these to load real-world datasets (finance, healthcare, image metadata, etc.) and export predictions after modeling

.
5. Exploring and Understanding Data
Function	Usage	ML Application
df.head(n)	First n rows	Quick data inspection
df.tail(n)	Last n rows	Check data ending
df.shape	Rows, columns	Dataset size for splitting
df.info()	Data types & nulls	Spot missing data, memory usage
df.describe()	Stats summary (count, std, etc.)	Detect outliers, feature ranges
df.columns	List of columns	Quick structure check

Example:

python
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

6. Data Selection, Indexing, and Slicing

    Single column: df['salary'] or df.salary

    Multiple columns: df[['age', 'salary']]

    Row selection by index: df.iloc

    Row/column selection by label: df.loc[0, 'age']

    Slice rows: df[5:10]

    Conditional selection: df[df['salary'] > 55000]

Application in ML:
Use these to extract features, select target variables, or filter samples of interest for training

.
7. Cleaning and Preparing Data
Handling Missing Data

    Check for missing values:

python
df.isnull().sum()

Drop missing:

python
df_clean = df.dropna()

Fill missing:

    python
    df_filled = df.fillna(0)

Renaming, Dropping, and Replacing

    Rename columns:

python
df = df.rename(columns={'old_col': 'new_col'})

Drop columns or rows:

python
df = df.drop('col_to_drop', axis=1)
df = df.drop([0, 2], axis=0)  # drop rows by index

Replace values:

    python
    df['col'] = df['col'].replace(0, np.nan)

Why?
Real ML data is messy—cleaning & imputation are essential preprocessing steps

.
8. Feature Engineering

    Transform columns:

        Create new features: df['age_squared'] = df['age'] ** 2

        Apply functions to columns:

    python
    df['income_log'] = df['salary'].apply(np.log)

One-hot encode:

    Convert categorical to numeric for ML algorithms:

    python
    pd.get_dummies(df['city'])

Binning:

    Group continuous values:

        python
        pd.cut(df['age'], bins=[0, 20, 40, 60], labels=['youth','adult','senior'])

Application:
Feature engineering boosts model accuracy and enables algorithms to interpret real-world meaning

.
9. Aggregating, Grouping, and Pivoting Data
GroupBy

    Summary by category:

python
df.groupby('department')['salary'].mean()

Multiple operations:

    python
    df.groupby('company').agg({'salary': ['mean', 'std'], 'age': 'median'})

Pivot Tables

    Like in Excel; summarize structured data:

    python
    pd.pivot_table(df, index='company', columns='gender', values='salary', aggfunc='mean')

ML Use:
Aggregate or restructure data to engineer more features, summarize trends, or generate custom inputs

.
10. Merging, Joining, and Concatenating DataFrames

Combine multiple tables—a frequent ML need (merging features, labels, external sources):

    Concatenate (add rows):

python
df_new = pd.concat([df1, df2], ignore_index=True)

Merge (add columns by key):

python
merged = pd.merge(df1, df2, how='inner', on='user_id')

Join (add columns by index):

    python
    df1.join(df2, lsuffix='_left', rsuffix='_right')

Why?
Join tables of features, enrich records for modeling, or stack batch predictions

.
11. Sorting and Ranking

    Sort by column value:

python
df.sort_values(by='salary', ascending=False)

Rank within columns:

    python
    df['score_rank'] = df['score'].rank()

Application:
Sort to identify best/worst samples, rank predictions, or structure data for visualization

.
12. Time Series in Pandas

    Convert to datetime:

python
df['date'] = pd.to_datetime(df['date'])

Set time as index & resample:

    python
    df = df.set_index('date')
    df.resample('M').mean()  # monthly average

ML Use:
Forecasting (sales, climate, stock), build lagged features, or time-window aggregations

.
13. Working with Text Data

    Lowercase a series:

    python
    df['text'] = df['text'].str.lower()

    Remove punctuation, split, extract patterns (with .str and regex functions).

ML Use:
For Natural Language Processing (NLP) preprocessing—tokenizing, cleaning, feature extraction for models

.
14. Visualizing Data with Pandas

    Plotting is built-in (matplotlib dependency):

python
df['salary'].plot(kind='hist')
df.plot(kind='scatter', x='age', y='salary')

Use Pandas Profiling for automatic reports (explores distributions, missing values, correlations)

:

    python
    import pandas_profiling
    profile = df.profile_report()
    profile.to_file("report.html")

Why?
Visuals help with feature selection, detecting outliers, and communicating findings

.
15. Advanced: Applying Functions and Lambda

    Apply a function to a series:

python
def to_grade(score): return 'pass' if score > 60 else 'fail'
df['result'] = df['score'].apply(to_grade)

Use lambda for quick, custom logic:

    python
    df['adjusted'] = df['salary'].apply(lambda x: x * 1.1)

16. Efficient Data Processing Tricks

    Vectorized operations for speed:

    python
    df['double'] = df['num'] * 2

    Avoid loops in Pandas, use .apply() or vector ops for efficient computation—vital with large ML datasets.

17. End-to-End ML Workflow Example

Here's a mini recipe for a typical ML data workflow:

python
import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv('train.csv')

# 2. Explore
print(df.head())
print(df.describe())

# 3. Clean
df = df.dropna(subset=['target'])              # Remove rows with missing target
df['feature'] = df['feature'].fillna(df['feature'].mean())   # Fill missing feature with mean

# 4. Feature engineering
df['feature_squared'] = df['feature']**2
df = pd.get_dummies(df, columns=['category'])

# 5. Split
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training: pass X_train, y_train to ML algorithm (scikit-learn, TensorFlow, PyTorch, etc.)

18. Real-World Use Cases

    Customer segmentation: Clean & group customer data to develop targeted marketing with clustering.

    Fraud detection: Aggregate and engineer features from transactional logs for anomaly detection.

    Image metadata preprocessing: Convert, merge, or rescale info for image ML tasks.

    Time series forecasting: Generate features with rolling means, lags, and resample to needed periods.

