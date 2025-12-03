#pragma once

#include <Eigen/Dense>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

class HungarianEigen
{
public:
    //Default constructor and destructor impl
    HungarianEigen() = default;
    ~HungarianEigen() = default;

    /**
     * @brief Solves the assignment problem for given resources and tasks. Cost matrix can be non-square
     * @param Eigen::MatrixXd& cost_matrix A square/non-square matrix containing the costs associated every resource and every task. cost: cost(i,j) = cost of assigning row i to column j
     * @param Eigen::VectorXi& assignment An empty Eigen Vector for assignments. assignment[i] = j or -1 (if unassigned in rectangular case)
     * @return double Resulting total cost of the assignment. This should be optimal
     */
    double solve(const Eigen::MatrixXd& cost_matrix,Eigen::VectorXi& assignment);
    
    /**
     * @brief Uses a completed assignment vector to return a vector of pairs matching a resource to an id
     * @param Eigen::VectorXi& Completed assignments which is produced by the solve method
     * @return The pairs of resource:task as a vector. 
     */
    std::vector<std::pair<size_t,size_t>> asVectorPairs(Eigen::VectorXi& completed_assignment);
};
