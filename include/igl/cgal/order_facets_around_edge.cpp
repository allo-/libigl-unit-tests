#include <test_common.h>

#include <iostream>

#include <vector>

#include <igl/cgal/order_facets_around_edge.h>
#include <igl/unique_edge_map.h>
#include <igl/readDMAT.h>
#include <igl/per_face_normals.h>

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
    igl::cgal::order_facets_around_edge(V, F, 3, 2, adj_faces_1, pivot, order_1);

    Eigen::VectorXi order_2;
    igl::cgal::order_facets_around_edge(V, F, 3, 2, adj_faces_2, pivot, order_2);

    ASSERT_EQ(adj_faces_1[order_1[0]], adj_faces_2[order_2[0]]);
    ASSERT_EQ(adj_faces_1[order_1[1]], adj_faces_2[order_2[1]]);
}
