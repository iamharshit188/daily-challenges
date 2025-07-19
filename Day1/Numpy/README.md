1. Setting Up NumPy

NumPy is the backbone of ML data processing in Python.

What for?

    Enables fast, efficient manipulation of multi-dimensional arrays essential for ML.

How to install and import:

python
# Install NumPy (if not already done)
!pip install numpy

# Import NumPy (always import as 'np' for convention)
import numpy as np

Where is it used?

    All ML workflows: data loading, feature engineering, models.

2. Basics: Creating and Inspecting Arrays

Purpose:

    Store, process, and inspect numerical data efficiently.

Creating Arrays

python
# 1D array
a = np.array([1, 2, 3], dtype='int32')
print(a)  # Output: [1 2 3]

# 2D array
b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
print(b)
# Output:
# [[9. 8. 7.]
#  [6. 5. 4.]]

Inspecting Array Properties

python
a.ndim      # Number of dimensions (1 for a)
b.shape     # Shape (rows, columns) ((2, 3) for b)
a.dtype     # Data type of elements (int32)
a.itemsize  # Bytes per element (4)
a.nbytes    # Total bytes (12 for 'a')
a.size      # Number of elements (3)

Real-life use:

    Images as 3D arrays, tabular data as 2D, time series as 2D (samples × steps)

3. Indexing, Slicing, and Modifying Arrays

Purpose:

    Fast access/modification of any data subset.

2D Array Example

python
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
# Access element at row 2, column 6
print(a[1, 5])  # Output: 13

# Get entire first row
print(a[0, :])  # Output: [1 2 3 4 5 6 7]

# Get third column
print(a[:, 2])  # Output: [ 3 10]

# Fancy slicing (step size)
print(a[0, 1:-1:2])  # Output: [2 4 6]

Modifying Elements

python
# Change an individual value
a[1, 5] = 20

# Change an entire column (broadcasting)
a[:, 2] = [1, 2]
print(a)

3D Array Example

python
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)

# Get specific element: first block, second row, second column
print(b[0, 1, 1])  # Output: 4

Real-life uses:

    Accessing a pixel region in an image, extracting a set of rows/features from data

4. Initializing Special Arrays

Purpose:

    Quickly create arrays with structure/patterns useful for ML and demos.

Examples

python
# 2x3 matrix of zeros
np.zeros((2,3))

# 4x2x2 array of ones
np.ones((4,2,2), dtype='int32')

# 2x2 matrix of all 99s
np.full((2,2), 99)

# Same shape as another array, filled with 4
np.full_like(a, 4)

# 4x2 array of random decimals
np.random.rand(4,2)

# 3x3 array of random integers between -4 and 7
np.random.randint(-4,8, size=(3,3))

# 5x5 identity matrix
np.identity(5)

# Repeat an array row-wise
arr = np.array([[1,2,3]])
np.repeat(arr, 3, axis=0)

Real-World ML Uses

    Random arrays: Initialize model weights or simulate test data

    Identity: Linear algebra, regularization, and transformations

    Patterns: Masks for image segmentation

5. Array Copying

Why is this important?

    Prevents accidental changes to original data (critical in pipelines).

python
a = np.array([1,2,3])
b = a.copy()
b[0] = 100
print(a)  # Output: [1 2 3] (unchanged)

Tip: Never assign with b = a if you want to keep the original intact.
6. Mathematics and Array Operations

Purpose:

    Enables fast, vectorized math for efficient data processing.

Elementwise Operations

python
a = np.array([1,2,3,4])

a + 2      # array([3, 4, 5, 6])
a * 2      # array([2, 4, 6, 8])
a ** 2     # array([ 1,  4,  9, 16])
a / 2      # array([0.5, 1., 1.5, 2.])
a - 2      # array([-1, 0, 1, 2])

# Add arrays of same shape
b = np.array([1, 0, 1, 0])
a + b      # array([2, 2, 4, 4])

Universal Functions

python
np.cos(a)  # Cosine of each element

More math routines: See NumPy and SciPy docs.

Real-life uses:

    Feature scaling, normalization, custom feature engineering in ML

7. Linear Algebra

Purpose:

    ML models depend on fast matrix computations.

Matrix Multiplication

python
a = np.ones((2, 3))            # shape (2, 3)
b = np.full((3, 2), 2)         # shape (3, 2)
np.matmul(a, b)                # shape (2, 2); Output: array of 6s

Determinant

python
c = np.identity(3)
np.linalg.det(c)               # Output: 1.0

Other functions:

    Eigenvalues, inverse, SVD, trace (see NumPy’s linalg module)

Real-world ML:

    Training deep nets (matrix mult)

    PCA (eigenvalues)

    System simulations and predictions

8. Statistics

Purpose:

    Summarize and analyze data sets for ML/EDA.

Usage

python
stats = np.array([[1,2,3],[4,5,6]])

np.min(stats)                  # Minimum value: 1
np.max(stats, axis=1)          # Max per row: [3, 6]
np.sum(stats, axis=0)          # Column-wise sum: [5, 7, 9]
np.mean(stats)                 # Mean of all elements
np.std(stats)                  # Standard deviation

Real-world ML:

    Data normalization, finding outliers, summarizing statistics before training

9. Reshaping and Stacking Arrays

Purpose:

    Prepare and combine data to match expected input formats for ML models.

Reshaping

python
before = np.array([[1,2,3,4],[5,6,7,8]])
after = before.reshape((4,2)) # Valid (8 elements)

Beware:

python
# after = before.reshape((2,3))  # Error: cannot reshape 8 elements into (2,3)

Stacking

python
# Vertical stacking (new rows)
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])
np.vstack([v1, v2, v1, v2])

# Horizontal stacking (new columns)
h1 = np.ones((2,4))
h2 = np.zeros((2,2))
np.hstack((h1,h2))

Real-life uses:

    Merging feature vectors, formatting tensors for DL models

10. Loading Data from Files

Purpose:

    Real datasets are usually in CSV or text files.

python
filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astype('int32')  # Type conversion if needed
print(filedata)

Pro-tip: Handle missing data and choose the right dtype for ML compatibility.
11. Boolean Masking and Advanced Indexing

Purpose:

    Fast filtering, selection, or modification based on conditions.

Example

python
# Create mask for elements not between 50 and 100
mask = ~((filedata > 50) & (filedata < 100))
print(mask)

# Use mask to filter data
filtered = filedata[mask]

Other tricks:

    Multiple conditions: (a > 5) & (a < 10)

    np.any(mask, axis=0) or np.all(mask, axis=1) for logic

    Assign/tweak only data meeting condition: filedata[filedata < 0] = 0

Real-world ML:

    Remove outliers, select high-value customers, build labels for supervised learning

Tips for Applied ML with NumPy

    Always check array shapes with .shape before math or stacking.

    Use .copy() to keep originals when experimenting.

    Vectorized computations (no Python for loops!) are much, much faster.

    NumPy feeds directly into all other ML libraries—Pandas, scikit-learn, TensorFlow, PyTorch.

