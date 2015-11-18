#include <test_common.h>
#include <Eigen/Dense>

#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

namespace RemeshSelfIntersectionsHelper {
    template<
        typename DerivedV,
        typename DerivedF >
    void assert_no_self_intersections(
            const Eigen::PlainObjectBase<DerivedV>& V,
            const Eigen::PlainObjectBase<DerivedF>& F) {

        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;

        MatrixXe VV;
        Eigen::MatrixXi FF, IF;
        Eigen::VectorXi J, IM;
        igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
        param.detect_only = true;
        igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);
        ASSERT_EQ(0, IM.rows());
    }

    template<
        typename DerivedV,
        typename DerivedF>
    void assert_no_degenerated_faces(
            const Eigen::PlainObjectBase<DerivedV>& V,
            const Eigen::PlainObjectBase<DerivedF>& F) {
        const size_t num_vertices = V.rows();
        const size_t num_faces = F.rows();
        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;
        std::vector<K::Point_3> points;
        for (size_t i=0; i<num_vertices; i++) {
            K::Point_3 p(V(i,0), V(i,1), V(i,2));
            points.push_back(p);
        }
        for (size_t i=0; i<num_faces; i++) {
            const Eigen::Vector3i f = F.row(i).eval();
            ASSERT_FALSE(CGAL::collinear(
                        points[f[0]], points[f[1]], points[f[2]]));
        }
    }

    TEST(RemeshSelfIntersections, CubeWithFold) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        test_common::load_mesh("cube_with_fold.ply", V, F);

        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;

        MatrixXe VV;
        Eigen::MatrixXi FF, IF;
        Eigen::VectorXi J, IM;
        igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
        igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);

        assert_no_degenerated_faces(VV, FF);
        assert_no_self_intersections(VV, FF);
    }

    TEST(RemeshSelfIntersections, TwoCubes) {
        Eigen::MatrixXd V1;
        Eigen::MatrixXi F1;
        test_common::load_mesh("cube.obj", V1, F1);

        Eigen::MatrixXd V2(V1);
        V2.col(0) = V2.col(0) + Eigen::MatrixXd::Ones(V1.rows(), 1);
        Eigen::MatrixXi F2 = F1.array() + V1.rows();

        Eigen::MatrixXd V(V1.rows() + V2.rows(), 3);
        Eigen::MatrixXi F(F1.rows() + F2.rows(), 3);
        V << V1, V2;
        F << F1, F2;

        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;

        MatrixXe VV;
        Eigen::MatrixXi FF, IF;
        Eigen::VectorXi J, IM;
        igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
        igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);

        assert_no_degenerated_faces(VV, FF);
        assert_no_self_intersections(VV, FF);
    }

    TEST(RemeshSelfIntersections, TwoBoxesStacked) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        test_common::load_mesh("two-boxes-bad-self-union.ply", V, F);

        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;

        MatrixXe VV;
        Eigen::MatrixXi FF, IF;
        Eigen::VectorXi J, IM;
        igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
        igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);

        assert_no_degenerated_faces(VV, FF);
        // TODO: self intersection detection uses double for CGAL::Box.  Need to
        // change that.
        //assert_no_self_intersections(VV, FF);
    }

    TEST(RemeshSelfIntersections, DuplicatedFaces) {
        Eigen::MatrixXd V(3, 3);
        V << 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0;
        Eigen::MatrixXi F(3, 3);
        F << 0, 1, 2,
             1, 0, 2,
             2, 0, 1;

        typedef CGAL::Exact_predicates_exact_constructions_kernel K;
        typedef Eigen::Matrix<K::FT, Eigen::Dynamic, Eigen::Dynamic> MatrixXe;

        MatrixXe VV;
        Eigen::MatrixXi FF, IF;
        Eigen::VectorXi J, IM;
        igl::copyleft::cgal::RemeshSelfIntersectionsParam param;
        igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);

        assert_no_degenerated_faces(VV, FF);
        assert_no_self_intersections(VV, FF);
        ASSERT_EQ(3, FF.rows());
    }
}
