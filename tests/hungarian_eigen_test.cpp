#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <set>
#include <memory>
#include <chrono>

#include "hungarian_eigen.hpp"


class HungarianEigenTest : public testing::Test
{
protected:
    void SetUp() override
    {
        solver_ = std::make_unique<HungarianEigen>();
    }

    std::unique_ptr<HungarianEigen> solver_;
    
    inline std::pair<double,std::chrono::milliseconds> helpSolve(Eigen::MatrixXd&cost , Eigen::VectorXi &assignment)
    {
        auto start = std::chrono::high_resolution_clock::now();
        double total_cost = solver_->solve(cost, assignment);
        auto finish   = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);

        return {total_cost,duration};
    }
};

//////////////////////////////////////////////////////////////////////
//////////////////Basic Functionality Tests///////////////////////////
//////////////////////////////////////////////////////////////////////
TEST_F(HungarianEigenTest,SquareMatrix_SimpleAssignment_2x2)
{
    Eigen::MatrixXd cost(2,2); 
    cost << 4,1, 
            2,3;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{1,0}; //ToDo
    double expected_total_cost{3.0}; //ToDo
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());

}

TEST_F(HungarianEigenTest,SquareMatrix_SimpleAssignment_3x3)
{
    Eigen::MatrixXd cost(3,3); 
    cost << 4,7,5, 
            2,5,6,
            7,3,4;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{2,0,1};
    double expected_total_cost{10.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,SquareMatrix_SimpleAssignment_4x4)
{
    Eigen::MatrixXd cost(4,4); 
    cost << 4,7,5,12,
            2,5,6,8,
            3,4,26,2,
            4,14,33,7;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{2,1,3,0};
    double expected_total_cost{16.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest, SquareMatrix_SimpleAssignment_5x5)
{
    Eigen::MatrixXd cost(5, 5);
    cost << 7, 53, 183, 439, 863,
            497, 383, 563, 79, 973,
            287, 63, 343, 169, 583,
            627, 343, 773, 959, 943,
            767, 473, 103, 699, 303;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,3,2,1,4};
    double expected_total_cost{1075.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest, SquareMatrix_Mixed_Infeasible_5x5)
{
    Eigen::MatrixXd cost(5,5);
    constexpr double INF = 1e12;
    cost << 3,   INF, 8,   INF, 2,
            INF, 5,   INF, 4,   INF,
            7,   INF, 1,   INF, INF,
            INF, 2,   INF, 6,   INF,
            4,   INF, INF, INF, 9;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{4,3,2,1,0};
    double expected_total_cost{13.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

//////////////////////////////////////////////////////////////////////
//////////////////Non Square Tests////////////////////////////////////
//////////////////////////////////////////////////////////////////////
TEST_F(HungarianEigenTest,Rectangular_Wide_3x5)
{
    Eigen::MatrixXd cost(3,5); 
    cost << 9,2,7,3,4,
            6,4,3,7,5,
            5,8,1,6,3;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{3,1,2}; //1,2,4 is also possible
    double expected_total_cost{8.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);//  
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Rectangular_Tall_5x3)
{
    Eigen::MatrixXd cost(5,3); 
    cost << 4,1,3,
            2,0,5,
            3,2,2,
            9,1,7,
            6,3,5;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{-1,0,2,1,-1};  //can also be 1 0 2 -1 -1 
    double expected_total_cost{5.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Rectangular_Mixed_Infeasible_3x5)
{
    Eigen::MatrixXd cost(3,5); 
    constexpr double INF = 1e12;
    cost << 4,   INF,  7,   2,   INF,
            INF, 3,    INF, 8,   1,
            6,   5,    INF, INF, 4;

    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{3,4,1}; //1,2,4 is also possible
    double expected_total_cost{8.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);//  
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Rectangular_Mixed_Infeasible_5x3)
{
    Eigen::MatrixXd cost(5,3); 
    constexpr double INF = 1e12;
    cost << INF, 2,   5,
            3,   INF, 4,
            INF, INF, 1,
            7,   6,   INF,
            1,   INF, INF;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{1, -1, 2, -1, 0};  //can also be 1 0 2 -1 -1 
    double expected_total_cost{4.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Rectangular_SingleRow_1x5)
{
    Eigen::MatrixXd cost(1,5); 
    cost << 8,3,5,7,2;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{4};
    double expected_total_cost{2.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}


TEST_F(HungarianEigenTest,Rectangular_SingleColumn_5x1)
{
    Eigen::MatrixXd cost(5,1); 
    cost << 8,
            3,
            5,
            7,
            2;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{-1,-1,-1,-1,0};
    double expected_total_cost{2.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i], static_cast<int>(num_tasks)) << "(" << assignment[i] << "<" << num_tasks << ")"; //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],static_cast<int>(num_tasks)); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

//////////////////////////////////////////////////////////////////////
//////////////////Edge Case Tests/////////////////////////////////////
//////////////////////////////////////////////////////////////////////
TEST_F(HungarianEigenTest,Edge_AllEqual_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 5,5,5,5,
            5,5,5,5,
            5,5,5,5,
            5,5,5,5;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,2,3}; 
    double expected_total_cost{20.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_RepeatedRows_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 5,1,9,3,
            5,1,9,3,
            2,7,1,8,
            6,4,2,1;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{1,0,2,3}; // 0,1,2,3 is also possible
    double expected_total_cost{8.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_RepeatedColumns_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 4,4,1,9,
            3,3,2,8,
            7,7,9,1,
            6,6,3,5;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{2,1,3,0}; //0,1,3,2 is also valid 
    double expected_total_cost{11.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_RowOfZeros_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 0,0,0,0,
            100,50,70,90,
            80,60,40,20,
            5,5,5,5;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,3,2};
    double expected_total_cost{75.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_ColumnOfZeros_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 0,100,100,100,
            0,90,80,70,
            0,60,50,40,
            0,30,20,10;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,3,2}; //0 1 2 3 also a possible assignment
    double expected_total_cost{150.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_SingleZero_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 9,9,9,9,
            9,0,9,9,
            9,9,9,9,
            9,9,9,9;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,2,3}; 
    double expected_total_cost{27.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_DiagonalVsAntiDiagonal_5x5)
{
    Eigen::MatrixXd cost(5,5);
    cost << 9,9,9,9,0,
            9,9,9,0,9,
            9,9,0,9,9,
            9,0,9,9,9,
            0,9,9,9,9;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{4,3,2,1,0};
    double expected_total_cost{0.0};
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Edge_LargeDynamicRange_5x5)
{
    Eigen::MatrixXd cost(5,5);
    cost << 1e6,2,300,400,500,
            200,1e6,300,400,500,
            300,200,1e6,400,500,
            400,300,200,1e6,500,
            500,400,300,200,1e6;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{1,0,4,2,3}; 
    double expected_total_cost{1102.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

//////////////////////////////////////////////////////////////////////
//////////////////Degenerate Cases////////////////////////////////////
//////////////////////////////////////////////////////////////////////
TEST_F(HungarianEigenTest,Degenerate_Checkerboard_6x6)
{
    Eigen::MatrixXd cost(6,6);
    cost << 0,9,0,9,0,9,
            9,0,9,0,9,0,
            0,9,0,9,0,9,
            9,0,9,0,9,0,
            0,9,0,9,0,9,
            9,0,9,0,9,0;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,2,3,4,5}; 
    double expected_total_cost{0.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Degenerate_Stripe_5x5)
{
    Eigen::MatrixXd cost(5,5);
    cost << 9,1,9,1,9,
            9,1,9,1,9,
            9,1,9,1,9,
            9,1,9,1,9,
            9,1,9,1,9;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{0,1,2,3,4,5};
    double expected_total_cost{29.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Degenerate_CrossZero_5x5)
{
    Eigen::MatrixXd cost(5,5);
    cost << 9,9,0,9,9,
            9,9,0,9,9,
            0,0,0,0,0,
            9,9,0,9,9,
            9,9,0,9,9;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{4,2,3,1,0};
    double expected_total_cost{27.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

TEST_F(HungarianEigenTest,Degenerate_Step5Forcing_4x4)
{
    Eigen::MatrixXd cost(4,4);
    cost << 5,4,3,2,
            4,5,4,3,
            3,4,5,4,
            2,3,4,5;
    Eigen::VectorXi assignment;
    
    // Num Agents expected
    size_t num_agents = static_cast<size_t>(cost.rows());
    // Num Tasks expected
    size_t num_tasks = static_cast<size_t>(cost.cols());
    // Num Unallocated Agents expected. This happens when there are more agents than tasks
    size_t num_unallocated_agents = static_cast<int>(num_agents) - static_cast<int>(num_tasks)  > 0 ? num_agents - num_tasks  : 0;
    // Num Unallocated Tasks expected. This happens when there are more tasks than agents
    size_t num_unallocated_tasks =  static_cast<int>(num_tasks)  - static_cast<int>(num_agents) > 0 ? num_tasks  - num_agents : 0;

    std::vector<int> expected_allocs{2,3,0,1}; 
    double expected_total_cost{12.0}; 
    
    // Do the solving, not that cost and assignment will be modified
    auto [total_cost,duration] = helpSolve(cost,assignment);

    // Test for total cost correctness against expected total cosr
    EXPECT_DOUBLE_EQ(total_cost,expected_total_cost) << "Total Cost (" << expected_total_cost << ") --> Computed Cost (" << total_cost << ")";
    
    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]);
        EXPECT_GE(assignment[i],-1); //greater than equal
        EXPECT_LT(assignment[i],num_tasks); //Less than
    }
    
    std::set<size_t> track_unallocated_agents;
    std::set<size_t> track_allocated_agents;
    for (size_t i = 0 ; i < num_agents ; ++i){track_unallocated_agents.insert(i);}
    
    std::set<size_t> track_unallocated_tasks;
    std::set<size_t> track_allocated_tasks;
    for (size_t i = 0 ; i < num_tasks ; ++i) {track_unallocated_tasks.insert(i);}

    for (Eigen::Index i = 0 ; i < assignment.size() ; ++i)
    {
        // Test for the correctness of allocation
        EXPECT_EQ(assignment[i],expected_allocs[static_cast<size_t>(i)]); //change according to assignment
        // test for the valid minimum value of assignments, which are task ids that can be >-1
        EXPECT_GE(assignment[i],-1); //greater than equal
        // test for the valid maximum value of assignment, which is the num_task excluded
        EXPECT_LT(assignment[i],num_tasks); //Less than

       if (assignment[i] >= 0)
       {
            // This means that agent i has a valid task assignment j = assignment[i]
            track_allocated_agents.insert(i);
            if (!track_unallocated_agents.erase(i))
            {
                FAIL() << "This cannot happen, suggests duplicate agents...";
            }
            track_allocated_tasks.insert(assignment[i]);
            if (!track_unallocated_tasks.erase(assignment[i]))
            {
                FAIL() << "This cannot happen, suggests duplicate assignments...";
            }
       }

       //ignore the else cus we dont have to do anything
    }

    // Test if total agent number is correct
    EXPECT_EQ(num_agents,track_allocated_agents.size() + track_unallocated_agents.size()) << "Total Expected Agents (" << num_agents << ") | Total Allocated Agents (" << track_allocated_agents.size() << ") + " << "Total Unallocated Agents" << track_unallocated_agents.size() << ")";
    EXPECT_EQ(num_agents - num_unallocated_agents , track_allocated_agents.size());
    EXPECT_EQ(num_unallocated_agents,track_unallocated_agents.size());
    
    // Test if the total task number is correct
    EXPECT_EQ(num_tasks,track_allocated_tasks.size() + track_unallocated_tasks.size()) << "Total Expected Tasks (" << num_tasks << ") | Total Allocated Tasks (" << track_allocated_tasks.size() << ") + " << "Total Unallocated Tasks" << track_unallocated_tasks.size() << ")";
    EXPECT_EQ(num_tasks - num_unallocated_tasks,track_allocated_tasks.size());
    EXPECT_EQ(num_unallocated_tasks,track_unallocated_tasks.size());
}

//////////////////////////////////////////////////////////////////////
//////////////////AsVectorPairs Test//////////////////////////////////
//////////////////////////////////////////////////////////////////////

TEST_F(HungarianEigenTest, AsVectorPairs_BasicConversion)
{
    Eigen::MatrixXd cost(3, 3);
    cost << 1, 100, 100,
            100, 2, 100,
            100, 100, 3;

    Eigen::VectorXi assignment;
    solver_->solve(cost, assignment);

    auto pairs = solver_->asVectorPairs(assignment);

    EXPECT_EQ(pairs.size(), 3);
    
    // Verify pairs match assignment
    for (size_t i = 0; i < pairs.size(); ++i) {
        EXPECT_EQ(pairs[i].first, static_cast<int>(i));
        EXPECT_EQ(pairs[i].second, assignment[i]);
    }
}

TEST_F(HungarianEigenTest, AsVectorPairs_WithUnassigned)
{
    // More rows than columns
    Eigen::MatrixXd cost(3, 2);
    cost << 1, 2,
            3, 5,
            4, 7;

    Eigen::VectorXi assignment;
    solver_->solve(cost, assignment);

    auto pairs = solver_->asVectorPairs(assignment);
    EXPECT_EQ(pairs.size(), 3);
    
    std::vector<int> expected {1,0,-1};

    for (const auto &[agent_id,task_id] : pairs)
    {
        EXPECT_LT(agent_id,assignment.size());
        EXPECT_EQ(task_id,assignment[agent_id]);
    }
}

// =============================================================================
// Exception throws
// =============================================================================

TEST_F(HungarianEigenTest, EmptyMatrix_ThrowsException)
{
    Eigen::MatrixXd cost(0, 0);
    Eigen::VectorXi assignment;

    EXPECT_THROW(solver_->solve(cost, assignment), std::invalid_argument);
}

TEST_F(HungarianEigenTest, ZeroRows_ThrowsException)
{
    Eigen::MatrixXd cost(0, 3);
    Eigen::VectorXi assignment;

    EXPECT_THROW(solver_->solve(cost, assignment), std::invalid_argument);
}

TEST_F(HungarianEigenTest, ZeroCols_ThrowsException)
{
    Eigen::MatrixXd cost(3, 0);
    Eigen::VectorXi assignment;

    EXPECT_THROW(solver_->solve(cost, assignment), std::invalid_argument);
}

TEST_F(HungarianEigenTest, NegativeCosts_ThrowsException)
{
    Eigen::MatrixXd cost(2, 2);
    cost << 1, -2,
            3, 4;

    Eigen::VectorXi assignment;

    EXPECT_THROW(solver_->solve(cost, assignment), std::invalid_argument);
}

TEST_F(HungarianEigenTest, ZeroCosts_Allowed)
{
    Eigen::MatrixXd cost(2, 2);
    cost << 0, 1,
            1, 0;

    Eigen::VectorXi assignment;
    double total_cost = solver_->solve(cost, assignment);

    EXPECT_DOUBLE_EQ(total_cost, 0.0);
}

TEST_F(HungarianEigenTest, AllSameCost)
{
    Eigen::MatrixXd cost(3, 3);
    cost.setConstant(5.0);

    Eigen::VectorXi assignment;
    double total_cost = solver_->solve(cost, assignment);

    EXPECT_DOUBLE_EQ(total_cost, 15.0);  // 3 * 5
    
    // All assignments should be unique
    std::set<int> assigned_cols;
    for (int i = 0; i < assignment.size(); ++i) {
        assigned_cols.insert(assignment[i]);
    }
    EXPECT_EQ(assigned_cols.size(), 3);
}

TEST_F(HungarianEigenTest, VeryLargeCosts)
{
    Eigen::MatrixXd cost(2, 2);
    cost << 1e12, 1,
            1, 1e12;

    Eigen::VectorXi assignment;
    double total_cost = solver_->solve(cost, assignment);

    EXPECT_DOUBLE_EQ(total_cost, 2.0);
    EXPECT_EQ(assignment[0], 1);
    EXPECT_EQ(assignment[1], 0);
}



// =============================================================================
// Determinism Tests
// =============================================================================

TEST_F(HungarianEigenTest, Deterministic_SameInputSameOutput)
{
    Eigen::MatrixXd cost(4, 4);
    cost << 10, 5, 13, 4,
            3, 9, 18, 7,
            15, 2, 8, 12,
            6, 14, 11, 1;

    std::vector<Eigen::VectorXi> results;
    
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXi assignment;
        solver_->solve(cost, assignment);
        results.push_back(assignment);
    }

    // All results should be identical
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(results[i].size(), results[0].size());
        for (int j = 0; j < results[i].size(); ++j) {
            EXPECT_EQ(results[i][j], results[0][j]);
        }
    }
}
