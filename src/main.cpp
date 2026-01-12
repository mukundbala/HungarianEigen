#include <iostream>
#include <Eigen/Dense>
#include "hungarian_eigen.hpp"
#include "chrono"

void run(const std::string& name, const Eigen::MatrixXd& M) 
{
    std::cout << "===== " << name << " =====\n";
    HungarianEigen solver;
    Eigen::VectorXi assignment;

    auto t0 = std::chrono::high_resolution_clock::now();
    double cost = solver.solve(M, assignment);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto result_vec = solver.asVectorPairs(assignment);

    std::cout << "Matrix:\n" << M << "\n";
    std::cout << "Cost: " << cost << "\n";
    std::cout<< "Time Taken: " << ms << "ms" << "\n";
    std::cout << "Assignments:" << "\n";
    for (const auto &[i,j] : result_vec)
    {
        std::cout << "Asset " << i << " matched to task " << j << "\n";
    }
    std::cout << "\n";
    
}

int main() {

    // 2×2
    Eigen::MatrixXd A(2,2);
    A << 4,1,
         2,3;
    run("2x2 basic", A);

    // 3×3
    Eigen::MatrixXd B(3,3);
    B << 9,2,7,
         6,4,3,
         5,8,1;
    run("3x3 classic", B);

    // Your earlier 4×4 example
    Eigen::MatrixXd C(4,4);
    C << 15,6,12,8,
         10,16,8,12,
         30,25,11,9,
         13,7,20,17;
    run("4x4 custom", C);

    // 4x4 easy diagonal (identity-like)
    Eigen::MatrixXd D(4,4);
    D << 1, 100, 100, 100,
         100, 1, 100, 100,
         100, 100, 1, 100,
         100, 100, 100, 1;
    run("4x4 diagonal optimum", D);

    // 4x4 all equal - infinite optimal assignments
    Eigen::MatrixXd E(4,4);
    E << 5,5,5,5,
         5,5,5,5,
         5,5,5,5,
         5,5,5,5;
    run("4x4 all-equal", E);

    // 5x5 large spread (tests numerical stability)
    Eigen::MatrixXd F(5,5);
    F << 1000, 2, 300, 400, 500,
         200, 1000, 300, 400, 500,
         300, 200, 1000, 400, 500,
         400, 300, 200, 1000, 500,
         500, 400, 300, 200, 1000;
    run("5x5 large-spread diagonal", F);

    // 3×5 wide matrix (more tasks than agents)
    Eigen::MatrixXd G(3,5);
    G << 9, 2, 7, 3, 4,
         6, 4, 3, 7, 5,
         5, 8, 1, 6, 3;
    run("3x5 wide", G);

    // 6×3 tall matrix (more agents than tasks)
    Eigen::MatrixXd H(6,3);
    H << 4, 1, 3,
         2, 0, 5,
         3, 2, 2,
         9, 1, 7,
         6, 3, 5,
         3, 4, 2;
    run("6x3 tall", H);

    // 4×4 step-5 forcing zig-zag structure
    Eigen::MatrixXd I(4,4);
    I << 7, 3, 6, 9,
         9, 7, 3, 6,
         6, 9, 7, 3,
         3, 6, 9, 7;
    run("4x4 step5-forcing zigzag", I);

    // 4×4 near-duplicate rows
    Eigen::MatrixXd J(4,4);
    J << 5, 1, 9, 3,
         5, 1, 9, 3,
         2, 7, 1, 8,
         6, 4, 2, 1;
    run("4x4 repeated rows", J);

    // 4×4 near-duplicate columns
    Eigen::MatrixXd K(4,4);
    K << 4, 4, 1, 9,
         3, 3, 2, 8,
         7, 7, 9, 1,
         6, 6, 3, 5;
    run("4x4 repeated columns", K);

    // 5×5 forcing multiple augmenting paths
    Eigen::MatrixXd L(5,5);
    L << 9, 1, 9, 1, 9,
         1, 9, 1, 9, 1,
         9, 1, 9, 1, 9,
         1, 9, 1, 9, 1,
         9, 1, 9, 1, 9;
    run("5x5 alternating stripes", L);

    // 4x4 all zeros but one large
    Eigen::MatrixXd M2(4,4);
    M2 << 0,0,0,0,
          0,0,0,0,
          0,0,0,0,
          0,0,0,1000;
    run("4x4 all-zero except one", M2);

    // 3×3 high asymmetry
    Eigen::MatrixXd N(3,3);
    N << 1,100,100,
         100,1,100,
         100,100,1;
    run("3x3 asymmetric diagonal", N);

    // 8×8 deterministic pseudorandom
    Eigen::MatrixXd O(8,8);
    O <<
        12, 7, 9, 7, 9, 8, 7, 6,
        8,  9, 6, 6, 6, 9, 7, 3,
        7,  17, 12, 14, 9, 16, 14, 12,
        8,  7, 13, 13, 15, 12, 14, 15,
        12, 10, 12, 13, 13, 9, 10, 8,
        6,  6, 6, 10, 12, 12, 11, 9,
        9,  10, 10, 9, 13, 12, 10, 9,
        10, 11, 11, 9, 8, 7, 9, 8;
    run("8x8 deterministic", O);
    
    // === DEGENERATE TEST SET ===

    // 1. Row of all zeros
    Eigen::MatrixXd P1(4,4);
    P1 << 0,0,0,0,
          100,50,70,90,
          80,60,40,20,
          5,5,5,5;
    run("Degenerate P1: row-of-zeros", P1);

    // 2. Column of all zeros
    Eigen::MatrixXd P2(4,4);
    P2 << 0,100,100,100,
          0,90,80,70,
          0,60,50,40,
          0,30,20,10;
    run("Degenerate P2: col-of-zeros", P2);

    // 3. All zero except one large value
    Eigen::MatrixXd P3(4,4);
    P3 << 0,0,0,0,
          0,0,0,0,
          0,0,0,0,
          0,0,0,999;
    run("Degenerate P3: almost-all-zero", P3);

    // 4. Identical rows
    Eigen::MatrixXd P4(5,5);
    P4 <<
        5,1,5,1,5,
        5,1,5,1,5,
        5,1,5,1,5,
        5,1,5,1,5,
        5,1,5,1,5;
    run("Degenerate P4: identical rows", P4);

    // 5. Identical columns
    Eigen::MatrixXd P5(5,5);
    P5 <<
        7,7,7,7,7,
        3,3,3,3,3,
        4,4,4,4,4,
        6,6,6,6,6,
        1,1,1,1,1;
    run("Degenerate P5: identical cols", P5);

    // 6. Uniform matrix (all equal)
    Eigen::MatrixXd P6(4,4);
    P6 <<
        10,10,10,10,
        10,10,10,10,
        10,10,10,10,
        10,10,10,10;
    run("Degenerate P6: uniform matrix", P6);

    // 7. Step-5 forcing structure
    Eigen::MatrixXd P7(4,4);
    P7 <<
        5,4,3,2,
        4,5,4,3,
        3,4,5,4,
        2,3,4,5;
    run("Degenerate P7: heavy step5", P7);

    // 8. Single zero
    Eigen::MatrixXd P8(4,4);
    P8 <<
        9,9,9,9,
        9,0,9,9,
        9,9,9,9,
        9,9,9,9;
    run("Degenerate P8: single-zero", P8);

    // 9. Zero diagonal, large off-diagonal
    Eigen::MatrixXd P9(6,6);
    P9 <<
        0,1000,1000,1000,1000,1000,
        1000,0,1000,1000,1000,1000,
        1000,1000,0,1000,1000,1000,
        1000,1000,1000,0,1000,1000,
        1000,1000,1000,1000,0,1000,
        1000,1000,1000,1000,1000,0;
    run("Degenerate P9: zero diagonal", P9);

    // 10. Anti-diagonal zeros
    Eigen::MatrixXd P10(6,6);
    P10 <<
        999,999,999,999,999,0,
        999,999,999,999,0,999,
        999,999,999,0,999,999,
        999,999,0,999,999,999,
        999,0,999,999,999,999,
        0,999,999,999,999,999;
    run("Degenerate P10: anti-diagonal zeros", P10);

    // 11. Rank-1 matrix (outer product)
    Eigen::VectorXd r(4), c(4);
    r << 1,2,3,4;
    c << 10,20,30,40;
    Eigen::MatrixXd P11 = r * c.transpose();
    run("Degenerate P11: rank-1 matrix", P11);

    // 12. Tall rectangular with zero column
    Eigen::MatrixXd P12(8,3);
    P12 <<
        0,5,7,
        0,4,9,
        0,6,10,
        0,8,12,
        0,2,8,
        0,9,11,
        0,3,6,
        0,1,5;
    run("Degenerate P12: tall with zero col", P12);

    // 13. Wide rectangular with zero row
    Eigen::MatrixXd P13(3,8);
    P13 <<
        0,0,0,0,0,0,0,0,
        9,8,7,6,5,4,3,2,
        2,3,4,5,6,7,8,9;
    run("Degenerate P13: wide with zero row", P13);

    // 14. Cross pattern of zeros
    Eigen::MatrixXd P14(5,5);
    P14 <<
        9,9,0,9,9,
        9,9,0,9,9,
        0,0,0,0,0,
        9,9,0,9,9,
        9,9,0,9,9;
    run("Degenerate P14: cross-zero", P14);

    // 15. Checkerboard zero pattern
    Eigen::MatrixXd P15(6,6);
    P15 <<
        0,9,0,9,0,9,
        9,0,9,0,9,0,
        0,9,0,9,0,9,
        9,0,9,0,9,0,
        0,9,0,9,0,9,
        9,0,9,0,9,0;
    run("Degenerate P15: checkerboard", P15);


    return 0;
}
