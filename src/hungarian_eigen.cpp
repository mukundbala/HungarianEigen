#include "hungarian_eigen.hpp"

double HungarianEigen::solve(const Eigen::MatrixXd& cost_matrix,Eigen::VectorXi& assignment)
{
    // use the Eigen::Index object to avoid annoying int comparison warnings
    using Index = Eigen::Index;

    // Defensive
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0)
    {
        throw std::invalid_argument("Hungarian: Empty Matrix");
    }

    // Defensive
    if ((cost_matrix.array() < 0).any())
    {
        throw std::invalid_argument("Hungarian: Negative Costs in cost matrix.");
    }

    // Catch degenerate cases, like single row or single col
    if (cost_matrix.cols() == 1)
    {
        /*
        Single column, means that theres num_agents == cost_matrix.rows(), the correct assignment
        would be the min cost of this column. There is a single task
        */
        assignment = Eigen::VectorXi::Constant(cost_matrix.rows(),-1);
        
        // The assignment cost is simply the min value
        Eigen::Index row;
        double assignment_cost = cost_matrix.col(0).minCoeff(&row);

        // populate the assignment
        assignment[row] = 0;
        
        // assignment cost
        return assignment_cost;
    }

    if (cost_matrix.rows() == 1)
    {
        /*
        Single row, means that there is on agent and num_tasks == cost_matrix.cols();
        */
       assignment = Eigen::VectorXi::Constant(1,-1);

       Eigen::Index col;
       double assignment_cost = cost_matrix.row(0).minCoeff(&col);

       assignment[0] = static_cast<int>(col);

       return assignment_cost;
    }

    Eigen::MatrixXd cost_matrix_padded;
    bool padding_used {false};
    constexpr double INFEASIBLE_COST = 1e12;

    if (cost_matrix.rows() != cost_matrix.cols())
    {
        padding_used = true;

        Eigen::Index new_sq_dim = std::max(cost_matrix.rows(),cost_matrix.cols());

        // Pad it    
        cost_matrix_padded = Eigen::MatrixXd::Constant(new_sq_dim,new_sq_dim,INFEASIBLE_COST);
        cost_matrix_padded.block(0,0,cost_matrix.rows(),cost_matrix.cols()) = cost_matrix;
    }


    // Get the total number of rows and columns.
    const Index nRows = padding_used ? cost_matrix_padded.rows() : cost_matrix.rows();
    const Index nCols = padding_used ? cost_matrix_padded.cols() : cost_matrix.cols();

    // Find the min dim
    const Index minDim = std::min(nRows, nCols);

    // Allocate here, whatever the user gives will be nuked and set to a vec of constants with -1
    assignment = Eigen::VectorXi::Constant(nRows, -1);

    // Working state matrices
    Eigen::MatrixXd dist = padding_used ? cost_matrix_padded : cost_matrix;
    Eigen::MatrixXi star  = Eigen::MatrixXi::Zero(nRows, nCols);
    Eigen::MatrixXi prime = Eigen::MatrixXi::Zero(nRows, nCols);

    // To keep track of the rows and columns we cover
    Eigen::VectorXi coverRow = Eigen::VectorXi::Zero(nRows);
    Eigen::VectorXi coverCol = Eigen::VectorXi::Zero(nCols);

    // Row reduction
    for (Index r = 0; r < nRows; ++r)
    {
        double min_row_val = dist.row(r).minCoeff();
        dist.row(r).array() -= min_row_val;
    }

    // Column reduction
    for (Index c = 0; c < nCols; ++c)
    {
        double min_c_val = dist.col(c).minCoeff();
        dist.col(c).array() -= min_c_val;
    }

    // Every row should have atleast 1 0 after the row and col reduction

    // Define eps for double comps. Epsilon is the smallest +ve value
    double POS_EPS = std::numeric_limits<double>::epsilon();

    // Do the initial assignments. This is probably not optimal
    for (Index r = 0; r < nRows; ++r)
    {
        // skip if the row has been assigned to a col
        if (coverRow[r]) continue;

        // For every row, we check each item and assign it to the row if the cost is 0
        for (Index c = 0; c < nCols; ++c)
        {
            if (dist(r, c) < POS_EPS && coverRow(r) == 0 && coverCol(c) == 0)
            {
                // starring essentially means we are matching r to c. 
                star(r, c) = 1;
                // r and c are now assigned so we cover the corresponding row and col up
                coverRow[r] = 1;
                coverCol[c] = 1;
                break;
            }
        }
    }
    // At this point, we have some tentative assignments

    // Reset covers so that we can improve out assignment
    coverRow.setZero();
    coverCol.setZero();

    // Helper lambdas
    auto coverColumnsWithStars = [&]() {
        coverCol.setZero();
        for (Index c = 0; c < nCols; ++c)
        {
            for (Index r = 0; r < nRows; ++r)
            {
                if (star(r, c) == 1)
                {
                    coverCol[c] = 1;
                    break;
                }
            }
        }
    };

    auto countCoveredColumns = [&]() -> Index {
        return static_cast<Index>((coverCol.array() == 1).count());
    };

    auto buildAssignment = [&]() {
        for (Index r = 0; r < nRows; ++r)
        {
            assignment[r] = -1;
            for (Index c = 0; c < nCols; ++c)
            {
                if (star(r, c) == 1)
                {
                    assignment[r] = static_cast<int>(c);
                    break;
                }
            }
        }
    };

    // Cover up all the columns that have been starred. This means that we cover up a task if its been assigned
    coverColumnsWithStars();

    // We keep improving until we have assigned all tasks
    while (countCoveredColumns() < minDim)
    {

        bool foundZero = true;
        Index zRow = -1, zCol = -1;

        while (true)
        {
            foundZero = false;
            for (Index r = 0; r < nRows; ++r)
            {
                if (coverRow[r]) continue;
                for (Index c = 0; c < nCols; ++c)
                {
                    if (!coverCol[c] && dist(r, c) < POS_EPS)
                    {
                        // Prime this zero
                        prime(r, c) = 1;

                        // Check if there's a starred zero in this row
                        Index starCol = -1;
                        for (Index cc = 0; cc < nCols; ++cc)
                        {
                            if (star(r, cc) == 1)
                            {
                                starCol = cc;
                                break;
                            }
                        }
                        
                        // If there are no starred 0s
                        if (starCol == -1)
                        {
                            
                            zRow = r;
                            zCol = c;
                            foundZero = true;
                            break;
                        }
                        else
                        {
                            // Cover this row and uncover the column of the star
                            coverRow[r] = 1;
                            coverCol[starCol] = 0;
                            foundZero = true;
                            break;
                        }
                    }
                    //else foundZero will be false
                }
                if (foundZero && zRow != -1) break;
            }

            if (foundZero && zRow != -1)
            {
                
                break;
            }

            if (!foundZero)
            {
                // Step 5: Adjust the matrix
                double h = std::numeric_limits<double>::infinity();
                for (Index r = 0; r < nRows; ++r)
                {
                    if (coverRow[r]) continue;
                    for (Index c = 0; c < nCols; ++c)
                    {
                        if (!coverCol[c])
                        {
                            double val = dist(r, c);
                            if (val < h) h = val;
                        }
                    }
                }

                // Add h to covered rows
                for (Index r = 0; r < nRows; ++r)
                {
                    if (coverRow[r])
                    {
                        dist.row(r).array() += h;
                    }
                }

                // Subtract h from uncovered columns
                for (Index c = 0; c < nCols; ++c)
                {
                    if (!coverCol[c])
                    {
                        dist.col(c).array() -= h;
                    }
                }
                // Then continue searching for uncovered zeros
            }
        } 

        ///////////
        // Build augmenting path from (zRow, zCol)
        ///////////
        std::vector<std::pair<Index, Index>> path;
        path.emplace_back(zRow, zCol);

        while (true)
        {
            // Find starred zero in the column of the last primed zero
            Index rStar = -1;
            Index cStar = path.back().second;
            for (Index r = 0; r < nRows; ++r)
            {
                if (star(r, cStar) == 1)
                {
                    rStar = r;
                    break;
                }
            }

            if (rStar == -1)
            {
                // No starred zero in this column -> end of path
                break;
            }

            path.emplace_back(rStar, cStar);

            // Find primed zero in the row of that starred zero
            Index cPrime = -1;
            for (Index c = 0; c < nCols; ++c)
            {
                if (prime(rStar, c) == 1)
                {
                    cPrime = c;
                    break;
                }
            }

            if (cPrime == -1)
            {
                // Should not happen if algorithm is correct
                break;
            }

            path.emplace_back(rStar, cPrime);
        }

        // Flip stars/primes on the augmenting path
        for (auto& rc : path)
        {
            Index r = rc.first;
            Index c = rc.second;
            if (star(r, c) == 1)
            {
                star(r, c) = 0;
            }
            else if (prime(r, c) == 1)
            {
                star(r, c) = 1;
            }
        }

        // Clear all primes and reset covers
        prime.setZero();
        coverRow.setZero();
        coverCol.setZero();

        // Cover columns with starred zeros again (Step 2a)
        coverColumnsWithStars();
        
    } // end main while

    // Build assignment from final starred zeros
    buildAssignment();

    if (padding_used)
    {
        assignment.conservativeResize(cost_matrix.rows());
        for (Index r = 0; r < cost_matrix.rows(); ++r)
        {
            if (assignment[r] >= cost_matrix.cols())
            {
                assignment[r] = -1;
            }
        }
    }

    // Compute cost using original matrix only
    double totalCost = 0.0;
    for (Index r = 0; r < cost_matrix.rows(); ++r)
    {
        int c = assignment[r];
        if (c >= 0 && c < cost_matrix.cols())
        {
            totalCost += cost_matrix(r, c);
        }
    }
    return totalCost;
}

std::vector<std::pair<int,int>> HungarianEigen::asVectorPairs(Eigen::VectorXi& completed_assignment)
{
    std::vector<std::pair<int,int>> out_vec;

    // Completed assignments is such that it shows completed_assignment[i] = j or -1 if unassigned
    // Whether i is the task or resource is up to the user to figure out when using the cost matrix
    // While this distinction doesnt matter for the solver, the user needs to be clear semantically on
    // what the resource is and what the task is

    size_t i = 0;
    for (const auto &j : completed_assignment)
    {
        out_vec.emplace_back(i,j);
        i++;
    }
    
    return out_vec;
}
