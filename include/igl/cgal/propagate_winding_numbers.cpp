#include <test_common.h>
#include <iostream>

#include <igl/cgal/propagate_winding_numbers.h>
#include <igl/writeOBJ.h>

TEST(PropagateWindingNumbers, Cube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);
    const size_t num_faces = F.rows();

    Eigen::MatrixXi W;
    ASSERT_TRUE(
        igl::cgal::propagate_winding_numbers_single_component(V, F, W));

    ASSERT_EQ(num_faces, W.rows());
    ASSERT_TRUE((W.col(0).array() == 0).all());
    ASSERT_TRUE((W.col(1).array() == 1).all());
    ASSERT_TRUE((W.col(0).array() < W.col(1).array()).all());
}

TEST(PropagateWindingNumbers, DoubleCube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube.obj", V, F);
    const size_t num_faces = F.rows();

    Eigen::MatrixXi W;
    ASSERT_TRUE(
            igl::cgal::propagate_winding_numbers_single_component(V, F, W));

    ASSERT_EQ(num_faces, W.rows());
    ASSERT_EQ(0, W.minCoeff());
    ASSERT_EQ(2, W.maxCoeff());
    ASSERT_TRUE((W.col(0).array() < W.col(1).array()).all());

    std::vector<Eigen::Vector3i> outer_hull;
    for (size_t i=0; i<num_faces; i++) {
        if (W(i,0)==0 && W(i,1)==1) {
            outer_hull.push_back(F.row(i));
        }
    }

    Eigen::MatrixXi outer_hull_F(outer_hull.size(), 3);
    for (size_t i=0; i<outer_hull.size(); i++) {
        outer_hull_F.row(i) = outer_hull[i].transpose();
    }

    igl::writeOBJ("double_cube_outer_hull.obj", V, outer_hull_F);
}

TEST(PropagateWindingNumbers, DoubleCube2) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube_2.obj", V, F);
    const size_t num_faces = F.rows();

    Eigen::MatrixXi W;
    ASSERT_TRUE(
            igl::cgal::propagate_winding_numbers_single_component(V, F, W));

    ASSERT_EQ(num_faces, W.rows());
    ASSERT_EQ(0, W.minCoeff());
    ASSERT_EQ(2, W.maxCoeff());
    ASSERT_TRUE((W.col(0).array() < W.col(1).array()).all());

    std::vector<Eigen::Vector3i> outer_hull;
    for (size_t i=0; i<num_faces; i++) {
        if (W(i,0)==0 && W(i,1)==1) {
            outer_hull.push_back(F.row(i));
        }
    }

    Eigen::MatrixXi outer_hull_F(outer_hull.size(), 3);
    for (size_t i=0; i<outer_hull.size(); i++) {
        outer_hull_F.row(i) = outer_hull[i].transpose();
    }

    igl::writeOBJ("double_cube_outer_hull_2.obj", V, outer_hull_F);
}

TEST(PropagateWindingNumbers, NestedCubeDifferentLabel) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("cube.obj", V1, F1);

    Eigen::MatrixXd V2 = V1 * 0.5;
    Eigen::MatrixXi F2 = F1.array() + V1.rows();

    Eigen::MatrixXd V(V1.rows() * 2, 3);
    Eigen::MatrixXi F(F1.rows() * 2, 3);
    V << V1, V2;
    F << F1, F2;

    Eigen::VectorXi labels(F.rows());
    labels << Eigen::VectorXi::Zero(F1.rows()),
              Eigen::VectorXi::Ones(F1.rows());

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_EQ(F.rows(), W.rows());
    ASSERT_EQ(4, W.cols());

    const size_t N = F1.rows();
    ASSERT_TRUE((W.block(0, 0, N, 1).array() == 0).all());
    ASSERT_TRUE((W.block(0, 1, N, 1).array() == 1).all());
    ASSERT_TRUE((W.block(0, 2, N, 1).array() == 0).all());
    ASSERT_TRUE((W.block(0, 3, N, 1).array() == 0).all());

    ASSERT_TRUE((W.block(N, 0, N, 1).array() == 1).all());
    ASSERT_TRUE((W.block(N, 1, N, 1).array() == 1).all());
    ASSERT_TRUE((W.block(N, 2, N, 1).array() == 0).all());
    ASSERT_TRUE((W.block(N, 3, N, 1).array() == 1).all());
}

TEST(PropagateWindingNumbers, NestedCubeSameLabel) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("cube.obj", V1, F1);

    Eigen::MatrixXd V2 = V1 * 0.5;
    Eigen::MatrixXi F2 = F1.array() + V1.rows();

    Eigen::MatrixXd V(V1.rows() * 2, 3);
    Eigen::MatrixXi F(F1.rows() * 2, 3);
    V << V1, V2;
    F << F1, F2;

    Eigen::VectorXi labels(F.rows());
    labels.setZero();

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_EQ(F.rows(), W.rows());
    ASSERT_EQ(2, W.cols());

    const size_t N = F1.rows();
    ASSERT_TRUE((W.block(0, 0, N, 1).array() == 0).all());
    ASSERT_TRUE((W.block(0, 1, N, 1).array() == 1).all());

    ASSERT_TRUE((W.block(N, 0, N, 1).array() == 1).all());
    ASSERT_TRUE((W.block(N, 1, N, 1).array() == 2).all());
}

TEST(PropagateWindingNumbers, NestedNonManifoldCubeDifferentLabel) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("non_manifold_double_cube.obj", V1, F1);

    Eigen::Vector3d center =
        (V1.colwise().maxCoeff() + V1.colwise().minCoeff()).eval() /2.0;

    const size_t num_vertices = V1.rows();
    const size_t num_faces = F1.rows();
    Eigen::MatrixXd V(2*num_vertices, 3);
    Eigen::MatrixXi F(2*num_faces, 3);

    V << (V1.rowwise()-center.transpose()) * 0.1,
         (V1.rowwise()-center.transpose()) * 10;
    F << F1, F1.array() + num_vertices;

    Eigen::VectorXi labels(num_faces*2);
    labels << Eigen::VectorXi::Zero(num_faces),
              Eigen::VectorXi::Ones(num_faces);

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_TRUE((W.block(0, 0, num_faces, 2).array() <= 2).all());
    ASSERT_TRUE((W.block(0, 2, num_faces, 2).array() == 2).all());

    ASSERT_TRUE((W.block(num_faces, 0, num_faces, 2).array() == 0).all());
    ASSERT_TRUE((W.block(num_faces, 2, num_faces, 2).array() <= 2).all());
}

TEST(PropagateWindingNumbers, NestedNonManifoldCubeSameLabel) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("non_manifold_double_cube.obj", V1, F1);

    Eigen::Vector3d center =
        (V1.colwise().maxCoeff() + V1.colwise().minCoeff()).eval() /2.0;

    const size_t num_vertices = V1.rows();
    const size_t num_faces = F1.rows();
    Eigen::MatrixXd V(2*num_vertices, 3);
    Eigen::MatrixXi F(2*num_faces, 3);

    V << (V1.rowwise()-center.transpose()) * 0.1,
         (V1.rowwise()-center.transpose()) * 10;
    F << F1, F1.array() + num_vertices;

    Eigen::VectorXi labels(num_faces*2);
    labels.setZero();

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_TRUE((W.block(0, 0, num_faces, 2).array() >= 2).all());
    ASSERT_TRUE((W.block(num_faces, 0, num_faces, 2).array() <= 2).all());
}

TEST(PropagateWindingNumbers, NestedNonManifoldCubeDifferentLabel2) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("non_manifold_double_cube_2.obj", V1, F1);

    Eigen::Vector3d center =
        (V1.colwise().maxCoeff() + V1.colwise().minCoeff()).eval() /2.0;

    const size_t num_vertices = V1.rows();
    const size_t num_faces = F1.rows();
    Eigen::MatrixXd V(2*num_vertices, 3);
    Eigen::MatrixXi F(2*num_faces, 3);

    V << (V1.rowwise()-center.transpose()) * 0.1,
         (V1.rowwise()-center.transpose()) * 10;
    F << F1, F1.array() + num_vertices;

    Eigen::VectorXi labels(num_faces*2);
    labels << Eigen::VectorXi::Zero(num_faces),
              Eigen::VectorXi::Ones(num_faces);

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_TRUE((W.block(0, 0, num_faces, 2).array() <= 2).all());
    ASSERT_TRUE((W.block(0, 2, num_faces, 2).array() == 2).all());

    ASSERT_TRUE((W.block(num_faces, 0, num_faces, 2).array() == 0).all());
    ASSERT_TRUE((W.block(num_faces, 2, num_faces, 2).array() <= 2).all());
}

TEST(PropagateWindingNumbers, NestedNonManifoldCubeSameLabel2) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("non_manifold_double_cube_2.obj", V1, F1);

    Eigen::Vector3d center =
        (V1.colwise().maxCoeff() + V1.colwise().minCoeff()).eval() /2.0;

    const size_t num_vertices = V1.rows();
    const size_t num_faces = F1.rows();
    Eigen::MatrixXd V(2*num_vertices, 3);
    Eigen::MatrixXi F(2*num_faces, 3);

    V << (V1.rowwise()-center.transpose()) * 0.1,
         (V1.rowwise()-center.transpose()) * 10;
    F << F1, F1.array() + num_vertices;

    Eigen::VectorXi labels(num_faces*2);
    labels.setZero();

    Eigen::MatrixXi W;
    igl::cgal::propagate_winding_numbers(V, F, labels, W);

    ASSERT_TRUE((W.block(0, 0, num_faces, 2).array() >= 2).all());
    ASSERT_TRUE((W.block(num_faces, 0, num_faces, 2).array() <= 2).all());
}

