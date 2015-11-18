#include <test_common.h>

#include <iostream>
#include <vector>

#include <igl/copyleft/cgal/order_facets_around_edge.h>

TEST(OrderFacetsAroundEdge, ConsistentOrdering) {
    Eigen::MatrixXd V(4, 3);
    V << 
        -0.05, -0.05, -0.15,
        -0.05,  0.15,  0.05,
        -0.05,  0.15, -0.15,
         0.15,  0.15, -0.15;

    Eigen::MatrixXi F(2, 3);
    F << 3,  2,  1,
         2,  3,  0;

    std::vector<int> adj_faces_1 = {2, -1};
    std::vector<int> adj_faces_2 = {-1, 2};

    Eigen::MatrixXd pivot(1, 3);
    pivot << 8.33333, 1.66667, -15;

    Eigen::VectorXi order_1;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 3, 2, adj_faces_1, pivot, order_1);

    Eigen::VectorXi order_2;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 3, 2, adj_faces_2, pivot, order_2);

    ASSERT_EQ(adj_faces_1[order_1[0]], adj_faces_2[order_2[0]]);
    ASSERT_EQ(adj_faces_1[order_1[1]], adj_faces_2[order_2[1]]);
}

TEST(OrderFacetsAroundEdge, PivotCoincideWithVertex) {
    Eigen::MatrixXd V(6, 3);
    V << 0.0, 0.0, 0.0, // 0
         1.0, 0.0, 0.0, // 1
         0.5, 1.0, 0.0, // 2
         0.5,-1.0, 0.0, // 3
         0.5, 0.0, 1.0, // 4
         0.5, 0.0,-1.0; // 5
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,
         0, 1, 3,
         0, 1, 4,
         0, 1, 5;

    std::vector<int> adj_faces = {-1, -2, -3, -4};

    // Pivot is the same as vertex 2.
    Eigen::MatrixXd pivot(1, 3);
    pivot << 0.5, 1.0, 0.0;

    Eigen::VectorXi order;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 0, 1, adj_faces, pivot, order);

    ASSERT_EQ(4, order.size());
    ASSERT_EQ(0, order[0]);
    ASSERT_EQ(3, order[1]);
    ASSERT_EQ(1, order[2]);
    ASSERT_EQ(2, order[3]);
}

TEST(OrderFacetsAroundEdge, DuplicatedFacets) {
    Eigen::MatrixXd V(6, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.5, 1.0, 0.0,
         0.5,-1.0, 0.0,
         0.5, 0.0, 1.0,
         0.5, 0.0,-1.0;
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,
         1, 0, 2,
         0, 1, 3,
         1, 0, 3;

    Eigen::MatrixXd pivot_1(1, 3);
    pivot_1 << 0.5, 0.0, 1.0;

    Eigen::VectorXi order_1;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 0, 1, {-1, 2, -3, 4},
            pivot_1, order_1);

    ASSERT_EQ(0, order_1[0]);
    ASSERT_EQ(1, order_1[1]);
    ASSERT_EQ(2, order_1[2]);
    ASSERT_EQ(3, order_1[3]);

    Eigen::MatrixXd pivot_2(1, 3);
    pivot_2 << 0.5, 0.0,-1.0;

    Eigen::VectorXi order_2;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 0, 1, {-1, 2, -3, 4},
            pivot_2, order_2);

    ASSERT_EQ(2, order_2[0]);
    ASSERT_EQ(3, order_2[1]);
    ASSERT_EQ(0, order_2[2]);
    ASSERT_EQ(1, order_2[3]);
}

TEST(OrderFacetsAroundEdge, DuplicatedFacetsCoplanarWithPivot) {
    Eigen::MatrixXd V(6, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.5, 1.0, 0.0,
         0.5,-1.0, 0.0,
         0.5, 0.0, 1.0,
         0.5, 0.0,-1.0;
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,
         1, 0, 2,
         0, 1, 3,
         1, 0, 3;

    Eigen::MatrixXd pivot_1(1, 3);
    pivot_1 << 0.5, 1.0, 0.0;

    Eigen::VectorXi order_1;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 0, 1, {-1, 2, -3, 4},
            pivot_1, order_1);

    ASSERT_EQ(0, order_1[0]);
    ASSERT_EQ(1, order_1[1]);
    ASSERT_EQ(2, order_1[2]);
    ASSERT_EQ(3, order_1[3]);

    Eigen::MatrixXd pivot_2(1, 3);
    pivot_2 << 0.5,-1.0, 0.0;

    Eigen::VectorXi order_2;
    igl::copyleft::cgal::order_facets_around_edge(V, F, 0, 1, {-1, 2, -3, 4},
            pivot_2, order_2);

    ASSERT_EQ(2, order_2[0]);
    ASSERT_EQ(3, order_2[1]);
    ASSERT_EQ(0, order_2[2]);
    ASSERT_EQ(1, order_2[3]);
}

TEST(OrderFacetsAroundEdge, PivotOnEdge) {
    Eigen::MatrixXd V(6, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.5, 1.0, 0.0,
         0.5,-1.0, 0.0,
         0.5, 0.0, 1.0,
         0.5, 0.0,-1.0;
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,
         1, 0, 2,
         0, 1, 3,
         1, 0, 3;

    Eigen::MatrixXd pivot(1, 3);
    pivot << 0.5, 0.0, 0.0;

    Eigen::VectorXi order;
    ASSERT_ANY_THROW(igl::copyleft::cgal::order_facets_around_edge(
            V, F, 0, 1, {-1, 2, -3, 4},
            pivot, order));
}

TEST(OrderFacetsAroundEdge, AllFacetsAreDegenerated) {
    Eigen::MatrixXd V(4, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.5, 0.0, 0.0,
         1.5, 0.0, 0.0;
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         1, 0, 3;

    Eigen::MatrixXd pivot(1, 3);
    pivot << 0.5, 1.0, 0.0;

    Eigen::VectorXi order;
    ASSERT_ANY_THROW(igl::copyleft::cgal::order_facets_around_edge(
            V, F, 0, 1, {-1, 2},
            pivot, order));
}

TEST(OrderFacetsAroundEdge, OneDegeratedFacet) {
    Eigen::MatrixXd V(4, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.5, 0.0, 0.1,
         1.5, 0.0, 0.0;
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         1, 0, 3;

    Eigen::MatrixXd pivot(1, 3);
    pivot << 0.5, 1.0, 0.0;

    Eigen::VectorXi order;
    ASSERT_ANY_THROW(igl::copyleft::cgal::order_facets_around_edge(
            V, F, 0, 1, {-1, 2},
            pivot, order));
}
