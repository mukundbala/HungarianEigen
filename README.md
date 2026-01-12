[![CI](https://github.com/mukundbala/HungarianEigen/actions/workflows/ci.yml/badge.svg?branch=main)](
https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml
)

# HungarianEigen
Implementation of Kuhn-Munkres Hungarian Matching Algorithm in modern C++ using Eigen. A cool project to understand the inner-workings for the Matching Algorithm.

Contains a hpp and cpp file. You can directly copy past it into your own code-base.

Todo: Benchmarking on much larger matrices
## Setting up
```bash
sudo apt install libeigen3-dev
git clone https://github.com/mukundbala/HungarianEigen.git
cd HungarianEigen
mkdir build && cd build

#Make and build
cmake ..
make -j
```

## Running the Script
```bash
#you must have built it following the previous step
cd HungarianEigen/build
./demo
```

## Usage
The main entry point is as follows, and can be found in the header file

```cpp
double solve(const Eigen::MatrixXd& cost_matrix,Eigen::VectorXi& assignment)
```
The 2 inputs are a completed cost matrix and an empty assignments vector. The cost matrix can be rectangular or sparse. The rows can refer to some resource, while the
columns can refer to some task. The semantic meaning of 
what rows and columns are is up to the user. However, this will treat it as rows - > resource, column - > task

## Tests

### Square Matrices

| Test Name                           | Matrix Size | Scenario               | Expected Behaviour                        |
| ----------------------------------- | ----------- | ---------------------- | ----------------------------------------- |
| `SquareMatrix_SimpleAssignment_2x2` | 2×2         | Simple unique minimum  | Correct optimal assignment and total cost |
| `SquareMatrix_SimpleAssignment_3x3` | 3×3         | Small square matrix    | Correct optimal cost and valid matching   |
| `SquareMatrix_SimpleAssignment_4x4` | 4×4         | Larger square matrix   | Valid one-to-one assignment               |
| `SquareMatrix_SimpleAssignment_5x5` | 5×5         | Larger cost matrix     | Correct minimum cost selection            |
| `SquareMatrix_Mixed_Infeasible_5x5` | 5×5         | INF (infeasible edges) | Skips infeasible edges, optimal cost      |

### Rectangular Matrices

| Test Name                          | Matrix Size | Shape         | Scenario                 |
| ---------------------------------- | ----------- | ------------- | ------------------------ |
| `Rectangular_Wide_3x5`             | 3×5         | Wide          | More tasks than agents   |
| `Rectangular_Tall_5x3`             | 5×3         | Tall          | More agents than tasks   |
| `Rectangular_Mixed_Infeasible_3x5` | 3×5         | Wide          | Infeasible edges present |
| `Rectangular_Mixed_Infeasible_5x3` | 5×3         | Tall          | Infeasible edges present |
| `Rectangular_SingleRow_1x5`        | 1×5         | Single row    | One agent, many tasks    |
| `Rectangular_SingleColumn_5x1`     | 5×1         | Single column | Many agents, one task    |

### Edge Cases

| Test Name                         | Matrix Size | Edge Case                      |
| --------------------------------- | ----------- | ------------------------------ |
| `Edge_AllEqual_4x4`               | 4×4         | All costs identical            |
| `Edge_RepeatedRows_4x4`           | 4×4         | Duplicate rows                 |
| `Edge_RepeatedColumns_4x4`        | 4×4         | Duplicate columns              |
| `Edge_RowOfZeros_4x4`             | 4×4         | Entire row of zero cost        |
| `Edge_ColumnOfZeros_4x4`          | 4×4         | Entire column of zero cost     |
| `Edge_SingleZero_4x4`             | 4×4         | One zero entry                 |
| `Edge_DiagonalVsAntiDiagonal_5x5` | 5×5         | Competing diagonal optima      |
| `Edge_LargeDynamicRange_5x5`      | 5×5         | Large numeric range (1e6–1e12) |

### Degenerate Casess

| Test Name                     | Matrix Size | Pattern                     |
| ----------------------------- | ----------- | --------------------------- |
| `Degenerate_Checkerboard_6x6` | 6×6         | Alternating low/high costs  |
| `Degenerate_Stripe_5x5`       | 5×5         | Repeated column stripes     |
| `Degenerate_CrossZero_5x5`    | 5×5         | Cross of zeros              |
| `Degenerate_Step5Forcing_4x4` | 4×4         | Forces deep Hungarian steps |

### Assignment Conversion Utilities
| Test Name                       | Scenario                                          |
| ------------------------------- | ------------------------------------------------- |
| `AsVectorPairs_BasicConversion` | Converts assignment vector to (agent, task) pairs |
| `AsVectorPairs_WithUnassigned`  | Handles unassigned agents correctly               |

### Exceptions

| Test Name                       | Invalid Input          | Expected                       |
| ------------------------------- | ---------------------- | ------------------------------ |
| `EmptyMatrix_ThrowsException`   | 0×0 matrix             | Throws `std::invalid_argument` |
| `ZeroRows_ThrowsException`      | 0×N                    | Throws exception               |
| `ZeroCols_ThrowsException`      | N×0                    | Throws exception               |
| `NegativeCosts_ThrowsException` | Negative values        | Throws exception               |
| `ZeroCosts_Allowed`             | Zero values            | Valid solution                 |
| `AllSameCost`                   | Uniform cost           | Valid assignment               |
| `VeryLargeCosts`                | Extremely large values | Stable numerical behaviour     |

### Determinism

| Test Name                           | Purpose                                     |
| ----------------------------------- | ------------------------------------------- |
| `Deterministic_SameInputSameOutput` | Ensures solver is deterministic across runs |
