#include <test_common.h>
#include <Eigen/Dense>

#include <igl/copyleft/cgal/extract_cells.h>

TEST(ExtractCells, SingleCube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    Eigen::MatrixXi cells;
    const size_t num_cells =
        igl::copyleft::cgal::extract_cells(V, F, cells);

    ASSERT_EQ(F.rows(), cells.rows());
    ASSERT_EQ(2, cells.cols());
    ASSERT_EQ(2, num_cells);
}

TEST(ExtractCells, NestedCubes) {
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    test_common::load_mesh("cube.obj", V1, F1);

    Eigen::MatrixXd V2 = V1 * 0.5;
    Eigen::MatrixXi F2 = F1.array() + V1.rows();

    Eigen::MatrixXd V(V1.rows() + V2.rows(), 3);
    Eigen::MatrixXi F(F1.rows() + F2.rows(), 3);
    V << V1, V2;
    F << F1, F2;

    Eigen::MatrixXi cells;
    const size_t num_cells =
        igl::copyleft::cgal::extract_cells(V, F, cells);

    ASSERT_EQ(3, num_cells);
    ASSERT_NE(cells(0, 0), cells(0, 1));
    ASSERT_EQ(cells(0, 1), cells(F.rows()-1, 0));
    ASSERT_NE(cells(F.rows()-1, 0), cells(F.rows()-1, 1));
}

//TEST(ExtractCells, TetWithDuplicatedFaces) {
//    Eigen::MatrixXd V;
//    Eigen::MatrixXi F;
//    test_common::load_mesh("tet_with_duplicated_faces.ply", V, F);
//
//    Eigen::MatrixXi cells;
//    const size_t num_cells =
//        igl::copyleft::cgal::extract_cells(V, F, cells);
//
//    ASSERT_EQ(4, num_cells);
//}

//TEST(ExtractCells, TwoCubes) {
//    Eigen::MatrixXd V;
//    Eigen::MatrixXi F;
//    test_common::load_mesh("two-boxes-bad-self-union.ply", V, F);
//
//    Eigen::MatrixXd VV;
//    Eigen::MatrixXi FF, IF;
//    Eigen::VectorXi J, IM;
//
//    igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
//    igl::copyleft::cgal::remesh_self_intersections(V, F, param,
//        VV, FF, IF, J, IM);
//    std::for_each(FF.data(), FF.data() + FF.size(), [&](int& a){a = IM[a];});
//
//    Eigen::MatrixXi cells;
//    const size_t num_cells =
//        igl::copyleft::cgal::extract_cells(VV, FF, cells);
//
//}
