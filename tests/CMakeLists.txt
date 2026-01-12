find_package(GTest REQUIRED)

add_executable(hungarian_eigen_test
    hungarian_eigen_test.cpp
)

target_link_libraries(hungarian_eigen_test
    PRIVATE
        hungarian_eigen_lib
        GTest::gtest
        GTest::gtest_main
)

add_test(
    NAME hungarian_eigen_test
    COMMAND hungarian_eigen_test
)
