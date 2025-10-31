# Data Structures

## Overview

`@tangent.to/ds` provides flexible data structures for statistical and machine learning workflows. The library uses standard JavaScript arrays and objects, with optional wrappers for enhanced functionality.

## Core Data Types

### Arrays

Most functions accept standard JavaScript arrays:

```javascript
// 1D array (vector)
const vector = [1, 2, 3, 4, 5];

// 2D array (matrix)
const matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

// Feature matrix (samples × features)
const X = [
  [5.1, 3.5, 1.4],  // Sample 1
  [4.9, 3.0, 1.4],  // Sample 2
  [4.7, 3.2, 1.3]   // Sample 3
];

// Target vector
const y = [0, 0, 1];
```

### Matrix Operations

The `core.linalg` module wraps `ml-matrix` for linear algebra:

```javascript
import { core } from '@tangent.to/ds';

// Convert to Matrix object
const M = core.linalg.toMatrix([[1, 2], [3, 4]]);

// Matrix operations
const MT = M.transpose();
const inv = core.linalg.inverse(M);

// Solve linear system: Ax = b
const A = [[2, 1], [1, 3]];
const b = [4, 5];
const x = core.linalg.solveLeastSquares(A, b);
```

### Tables

The `core.table` module provides DataFrame-like functionality:

```javascript
import { core } from '@tangent.to/ds';

// Create table from arrays
const data = {
  name: ['Alice', 'Bob', 'Charlie'],
  age: [25, 30, 35],
  score: [85, 90, 95]
};

const table = core.table.fromObject(data);

// Access columns
const ages = core.table.column(table, 'age');  // [25, 30, 35]

// Filter rows
const adults = core.table.filter(table, row => row.age >= 30);

// Select columns
const subset = core.table.select(table, ['name', 'score']);
```

## Data Conversion

### Object to Array

```javascript
// Object with column arrays
const obj = {
  x: [1, 2, 3],
  y: [4, 5, 6]
};

// Convert to row-major array
const rows = Object.keys(obj)[0] ? 
  obj.x.map((_, i) => Object.values(obj).map(col => col[i])) :
  [];
// [[1, 4], [2, 5], [3, 6]]
```

### CSV-Like Data

```javascript
// Parse CSV-like string (simple case)
const csv = `x,y,z
1,2,3
4,5,6
7,8,9`;

const lines = csv.trim().split('\n');
const headers = lines[0].split(',');
const data = lines.slice(1).map(line => 
  line.split(',').map(Number)
);
```

## Data Validation

### Check Data Types

```javascript
import { core } from '@tangent.to/ds';

// Validate matrix dimensions
function isValidMatrix(X) {
  if (!Array.isArray(X) || X.length === 0) return false;
  const nCols = X[0].length;
  return X.every(row => Array.isArray(row) && row.length === nCols);
}

// Validate target vector
function isValidTarget(y, n) {
  return Array.isArray(y) && y.length === n;
}

// Use built-in guards
core.math.isFinite(value);  // Check for finite number
core.math.approxEqual(a, b, epsilon);  // Float comparison
```

### Handle Missing Values

```javascript
// Remove rows with missing values
function dropNA(X, y) {
  const valid = X.map((row, i) => ({
    x: row,
    y: y[i],
    valid: row.every(v => v !== null && v !== undefined && !isNaN(v))
  }));
  
  const cleaned = valid.filter(item => item.valid);
  return {
    X: cleaned.map(item => item.x),
    y: cleaned.map(item => item.y)
  };
}

// Impute with mean
function imputeMean(X) {
  const nCols = X[0].length;
  const means = [];
  
  for (let j = 0; j < nCols; j++) {
    const col = X.map(row => row[j]).filter(v => !isNaN(v));
    means[j] = col.reduce((a, b) => a + b, 0) / col.length;
  }
  
  return X.map(row => 
    row.map((val, j) => isNaN(val) ? means[j] : val)
  );
}
```

## Memory Management

### In-Place Operations

For large datasets, prefer in-place operations:

```javascript
// Bad: Creates new array
const scaled = X.map(row => row.map(val => val / max));

// Better: Modify in-place (if array is not reused)
for (let i = 0; i < X.length; i++) {
  for (let j = 0; j < X[i].length; j++) {
    X[i][j] /= max;
  }
}

// Best: Use library functions that handle this
const scaler = new ml.preprocessing.StandardScaler();
scaler.fit(X);
const XScaled = scaler.transform(X);
```

### Typed Arrays

For numerical computing with large arrays:

```javascript
// Convert to typed array for performance
const vector = new Float64Array([1, 2, 3, 4, 5]);

// Matrix as flattened typed array
const matrix = new Float64Array([
  1, 2, 3,
  4, 5, 6
]);
const nRows = 2, nCols = 3;

// Access element (i, j)
const value = matrix[i * nCols + j];
```

## Best Practices

1. **Use consistent shapes**: All samples should have the same number of features
2. **Validate inputs**: Check for NaN, Infinity, null before analysis
3. **Document dimensions**: Use comments like `// X: n × p matrix`
4. **Avoid mutation**: Create copies when modifying data used elsewhere
5. **Use library functions**: They handle edge cases and optimizations

## Common Patterns

### Train-Test Split

```javascript
import { ml } from '@tangent.to/ds';

const { XTrain, XTest, yTrain, yTest } = ml.validation.trainTestSplit(
  X, y, 
  { testSize: 0.3, shuffle: true }
);
```

### Feature Scaling

```javascript
import { ml } from '@tangent.to/ds';

const scaler = new ml.preprocessing.StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);
```

### One-Hot Encoding

```javascript
import { ml } from '@tangent.to/ds';

const categories = ['red', 'green', 'blue', 'red'];
const encoder = new ml.preprocessing.OneHotEncoder();
encoder.fit(categories);
const encoded = encoder.transform(categories);
// [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
```

## Next Steps

- Learn about [Statistical Analysis](./03-statistics.md)
- Explore [Machine Learning](./04-machine-learning.md)
- Check [Data Preprocessing](./08-validation.md)
