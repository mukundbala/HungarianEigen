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