#include <test_common.h>
#include <Eigen/Dense>
#include <vector>

#include <igl/extract_non_manifold_edge_curves.h>
#include <igl/cgal/remesh_self_intersections.h>
#include <igl/unique_edge_map.h>
#include <igl/writeOBJ.h>

TEST(ExtractNonManifoldEdgeCurves, SingleCube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(F, E, uE, EMAP, uE2E);

    std::vector<std::vector<size_t> > curves;
    igl::extract_non_manifold_edge_curves(F, EMAP, uE2E, curves);

    ASSERT_EQ(0, curves.size());
}

TEST(ExtractNonManifoldEdgeCurves, DoubleCube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube.obj", V, F);

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(F, E, uE, EMAP, uE2E);

    std::vector<std::vector<size_t> > curves;
    igl::extract_non_manifold_edge_curves(F, EMAP, uE2E, curves);

    ASSERT_EQ(1, curves.size());
    ASSERT_EQ(6, curves[0].size());
    for (auto uei : curves[0]) {
        ASSERT_GT(uE2E.size(), uei);
    }
}

TEST(ExtractNonManifoldEdgeCurves, DoubleCube2) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube_2.obj", V, F);

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(F, E, uE, EMAP, uE2E);

    std::vector<std::vector<size_t> > curves;
    igl::extract_non_manifold_edge_curves(F, EMAP, uE2E, curves);

    ASSERT_EQ(12, curves.size());
    for (auto& curve : curves) {
        for (auto uei : curve) {
            ASSERT_GT(uE2E.size(), uei);
        }
    }
}

TEST(ExtractNonManifoldEdgeCurves, DuplicatedFace) {
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;
    Eigen::MatrixXi F(3, 3);
    F << 0, 1, 2,
         2, 1, 0,
         1, 2, 0;

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<size_t> > uE2E;
    igl::unique_edge_map(F, E, uE, EMAP, uE2E);

    std::vector<std::vector<size_t> > curves;
    igl::extract_non_manifold_edge_curves(F, EMAP, uE2E, curves);

    ASSERT_EQ(1, curves.size());
    ASSERT_EQ(3, curves[0].size());
    for (auto uei : curves[0]) {
        ASSERT_GT(uE2E.size(), uei);
    }
}
