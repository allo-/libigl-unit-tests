#include <test_common.h>

#include <iostream>
#include <vector>

#include <igl/cgal/closest_facet.h>

TEST(ClosestFacet, SingleFacetQueryProjectToCorner) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.0, 0.0, 1.0,
         0.0, 0.0,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    ASSERT_TRUE((index.array() == 0).all());
    ASSERT_TRUE(orientation[0]);
    ASSERT_FALSE(orientation[1]);
}

TEST(ClosestFacet, SingleFacetQueryProjectToEdge) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.5, 0.0, 1.0,
         0.5, 0.0,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    ASSERT_TRUE((index.array() == 0).all());
    ASSERT_TRUE(orientation[0]);
    ASSERT_FALSE(orientation[1]);
}

TEST(ClosestFacet, SingleFacetQueryProjectToFace) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.1, 0.1, 1.0,
         0.1, 0.1,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    ASSERT_TRUE((index.array() == 0).all());
    ASSERT_TRUE(orientation[0]);
    ASSERT_FALSE(orientation[1]);
}

TEST(ClosestFacet, DuplicatedFacetQueryProjectToCorner) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         1, 0, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.0, 0.0, 1.0,
         0.0, 0.0,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    // The closest facet could be either one, so nothing to check.
    ASSERT_TRUE(orientation[0]);
    ASSERT_TRUE(orientation[1]);
}

TEST(ClosestFacet, DuplicatedFacetQueryProjectToEdge) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         1, 0, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.1, 0.0, 1.0,
         0.1, 0.0,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    // The closest facet could be either one, so nothing to check.
    ASSERT_TRUE(orientation[0]);
    ASSERT_TRUE(orientation[1]);
}

TEST(ClosestFacet, DuplicatedFacetQueryProjectToFace) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;

    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         1, 0, 2;

    Eigen::MatrixXd P(2, 3);
    P << 0.1, 0.1, 1.0,
         0.1, 0.1,-1.0;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(2, index.size());
    ASSERT_EQ(2, orientation.size());
    // The closest facet could be either one, so nothing to check.
    ASSERT_TRUE(orientation[0]);
    ASSERT_TRUE(orientation[1]);
}

TEST(ClosestFacet, OutsideCubeCheck) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::MatrixXd P(6, 3);
    P << 1.1, 0.0, 0.0,
        -1.1, 0.0, 0.0,
         0.0, 1.1, 0.0,
         0.0,-1.1, 0.0,
         0.0, 0.0, 1.1,
         0.0, 0.0,-1.1;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(6, index.size());
    ASSERT_EQ(6, orientation.size());
    ASSERT_TRUE((orientation.array() != 0).all());
}

TEST(ClosestFacet, InsideCubeCheck) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::MatrixXd P(6, 3);
    P << 0.1, 0.0, 0.0,
        -0.1, 0.0, 0.0,
         0.0, 0.1, 0.0,
         0.0,-0.1, 0.0,
         0.0, 0.0, 0.1,
         0.0, 0.0,-0.1;

    Eigen::VectorXi index, orientation;
    igl::cgal::closest_facet(V, F, P,
            index, orientation);

    ASSERT_EQ(6, index.size());
    ASSERT_EQ(6, orientation.size());
    ASSERT_TRUE((orientation.array() == 0).all());
}
