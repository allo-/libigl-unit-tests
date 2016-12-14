#include <test_common.h>
#include <igl/readPLY.h>

using namespace test_common;

TEST(readPLY, file_does_not_exist){
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readPLY(test_common::data_path("file_does_not_exist.ply"), V, F);
}

TEST(readPLY, cube){
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readPLY(test_common::data_path("cube.ply"), V, F);

    ASSERT_EQ(8, V.rows());
    ASSERT_EQ(12, F.rows());
}

TEST(readPLY, two_boxes_bad_self_union){
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readPLY(test_common::data_path("two-boxes-bad-self-union.ply"), V, F);

    ASSERT_EQ(486, V.rows());
    ASSERT_EQ(708, F.rows());
}
