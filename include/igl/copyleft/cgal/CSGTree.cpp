#include <test_common.h>

#include <igl/copyleft/cgal/CSGTree.h>
#include <igl/doublearea.h>

TEST(CSGTree, extrusion) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("extrusion.obj", V, F);
    igl::copyleft::cgal::CSGTree tree(V, F);
    igl::copyleft::cgal::CSGTree inter(tree, tree, "i"); // returns error

    Eigen::MatrixXd V2 = inter.cast_V<Eigen::MatrixXd>();
    Eigen::MatrixXi F2 = inter.F();

    ASSERT_EQ(V.rows(), V2.rows());
    ASSERT_EQ(F.rows(), F2.rows());
}

TEST(CSGTree, cube) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    test_common::load_mesh("cube.obj", V, F);

    igl::copyleft::cgal::CSGTree tree({V, F}, {V, F}, "u");
    Eigen::MatrixXd V2 = tree.cast_V<Eigen::MatrixXd>();
    Eigen::MatrixXi F2 = tree.F();

    ASSERT_EQ(V.rows(), V2.rows());
    ASSERT_EQ(F.rows(), F2.rows());
}

TEST(CSGTree, A_minus_B__union__A_intersect_B) {
    Eigen::MatrixXd VA, VB;
    Eigen::MatrixXi FA, FB;
    test_common::load_mesh("cube.obj", VA, FA);
    test_common::load_mesh("TinyTorus.obj", VB, FB);

    igl::copyleft::cgal::CSGTree tree{
        {{VA, FA}, {VB, FB}, "minus"},
        {{VA, FA}, {VB, FB}, "intersect"},
        "union"
    };

    Eigen::MatrixXd VC = tree.cast_V<Eigen::MatrixXd>();
    Eigen::MatrixXi FC = tree.F();

    Eigen::VectorXd area_A, area_C;
    igl::doublearea(VA, FA, area_A);
    igl::doublearea(VC, FC, area_C);

    ASSERT_FLOAT_EQ(area_A.sum(), area_C.sum());
}

TEST(CSGTree, A_xor_B___minus___A_union_B__minus__A_intersect_B) {
    Eigen::MatrixXd VA, VB;
    Eigen::MatrixXi FA, FB;
    test_common::load_mesh("cube.obj", VA, FA);
    test_common::load_mesh("decimated-knight.obj", VB, FB);

    igl::copyleft::cgal::CSGTree tree{
        {{VA, FA}, {VB, FB}, "xor"},
        {
            {{VA, FA}, {VB, FB}, "union"},
            {{VA, FA}, {VB, FB}, "intersect"},
            "minus"
        }, "minus"
    };
    Eigen::MatrixXd VC = tree.cast_V<Eigen::MatrixXd>();
    Eigen::MatrixXi FC = tree.F();

    ASSERT_EQ(0, VC.rows());
    ASSERT_EQ(0, FC.rows());
}
