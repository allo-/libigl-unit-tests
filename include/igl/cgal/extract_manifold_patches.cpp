#include <test_common.h>
#include <Eigen/Dense>
#include <vector>

#include <igl/extract_manifold_patches.h>
#include <igl/cgal/remesh_self_intersections.h>
#include <igl/unique_edge_map.h>
#include <igl/writeOBJ.h>

TEST(ExtractManifoldPatches, SingleCube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP, P;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(F, E, uE, EMAP, uE2E);
    igl::extract_manifold_patches(F, EMAP, uE2E, P);

    ASSERT_EQ(F.rows(), P.size());
    ASSERT_TRUE((P.array() == 0).all());
}

TEST(ExtractManifoldPatches, DoubleCube) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("cube.obj", V1, F1);

    Eigen::VectorXd offset(3);
    offset.setConstant(0.5);

    Eigen::MatrixXd V2 = V1.rowwise() + offset.transpose();
    Eigen::MatrixXi F2 = F1.array() + V1.rows();

    Eigen::MatrixXd V(V1.rows() + V2.rows(), 3);
    V << V1, V2;
    Eigen::MatrixXi F(F1.rows() + F2.rows(), 3);
    F << F1, F2;

    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF, IF;
    Eigen::VectorXi J, IM;

    igl::cgal::RemeshSelfIntersectionsParam param;
    igl::cgal::remesh_self_intersections(V, F, param,
        VV, FF, IF, J, IM);
    std::for_each(FF.data(), FF.data() + FF.size(), [&](int& a){a = IM[a];});

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP, P;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(FF, E, uE, EMAP, uE2E);
    size_t num_patches = igl::extract_manifold_patches(FF, EMAP, uE2E, P);

    ASSERT_EQ(FF.rows(), P.size());
    ASSERT_EQ(4, num_patches);
    ASSERT_EQ(3, P.maxCoeff());
}
