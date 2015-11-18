#include <test_common.h>
#include <Eigen/Core>
#include <igl/copyleft/cgal/peel_winding_number_layers.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/remove_unreferenced.h>

TEST(PeelWindingNumberLayers, Cube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::VectorXi W;
    size_t num_layers = igl::copyleft::cgal::peel_winding_number_layers(V, F, W);

    ASSERT_EQ(1, num_layers);
    ASSERT_TRUE((W.array() == 1).all());
}

TEST(PeelWindingNumberLayers, NestedCube) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("cube.obj", V1, F1);

    const size_t num_vertices = V1.rows();
    const size_t num_faces = F1.rows();
    const size_t k = 10;
    Eigen::MatrixXd V(num_vertices * k, 3);
    Eigen::MatrixXi F(num_faces * k, 3);
    for (size_t i=0; i<k; i++) {
        V.block(i*num_vertices, 0, num_vertices, 3) =
            V1.array() * (1.0 - 1.0 / double(k+1) * i);
        F.block(i*num_faces, 0, num_faces, 3) =
            F1.array() + num_vertices * i;
    }

    Eigen::VectorXi W;
    size_t num_layers = igl::copyleft::cgal::peel_winding_number_layers(V, F, W);

    ASSERT_EQ(k, num_layers);
    for (size_t i=0; i<k; i++) {
        ASSERT_TRUE((W.segment(i*num_faces, num_faces).array() == i+1).all());
    }
}

TEST(PeelWindingNumberLayers, TwoCubes) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("two-boxes-bad-self-union.ply", V, F);
    ASSERT_EQ(486, V.rows());
    ASSERT_EQ(708, F.rows());

    typedef CGAL::Exact_predicates_exact_constructions_kernel K;
    typedef K::FT Scalar;
    typedef Eigen::Matrix<Scalar,
            Eigen::Dynamic,
            Eigen::Dynamic> MatrixXe;

    MatrixXe Vs;
    Eigen::MatrixXi Fs, IF;
    Eigen::VectorXi J, IM;
    igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
    igl::copyleft::cgal::remesh_self_intersections(V, F, param, Vs, Fs, IF, J, IM);

    std::for_each(Fs.data(),Fs.data()+Fs.size(),
            [&IM](int & a){ a=IM(a); });
    MatrixXe Vt;
    Eigen::MatrixXi Ft;
    igl::remove_unreferenced(Vs,Fs,Vt,Ft,IM);
    const size_t num_faces = Ft.rows();

    Eigen::VectorXi W;
    size_t num_layers = igl::copyleft::cgal::peel_winding_number_layers(Vt, Ft, W);

    ASSERT_EQ(2, num_layers);
}

TEST(PeelWindingNumberLayers, NestedNonManifoldCubeDifferentLabel) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube.obj", V, F);

    Eigen::VectorXi W;
    size_t num_layers = igl::copyleft::cgal::peel_winding_number_layers(V, F, W);

    ASSERT_EQ(2, num_layers);
}

TEST(PeelWindingNumberLayers, NestedNonManifoldCubeDifferentLabel2) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("non_manifold_double_cube_2.obj", V, F);

    Eigen::VectorXi W;
    size_t num_layers = igl::copyleft::cgal::peel_winding_number_layers(V, F, W);

    ASSERT_EQ(2, num_layers);
}
