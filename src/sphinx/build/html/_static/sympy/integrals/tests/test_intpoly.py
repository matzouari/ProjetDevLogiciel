from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.core import S, Rational

from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
                                     polytope_integrate, point_sort,
                                     hyperplane_parameters, main_integrate3d,
                                     main_integrate, polygon_integrate,
                                     lineseg_integrate, integration_reduction,
                                     integration_reduction_dynamic, is_vertex)

from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z

from sympy.testing.pytest import slow


def test_decompose():
    assert decompose(x) == {1: x}
    assert decompose(x**2) == {2: x**2}
    assert decompose(x*y) == {2: x*y}
    assert decompose(x + y) == {1: x + y}
    assert decompose(x**2 + y) == {1: y, 2: x**2}
    assert decompose(8*x**2 + 4*y + 7) == {0: 7, 1: 4*y, 2: 8*x**2}
    assert decompose(x**2 + 3*y*x) == {2: x**2 + 3*x*y}
    assert decompose(9*x**2 + y + 4*x + x**3 + y**2*x + 3) ==\
        {0: 3, 1: 4*x + y, 2: 9*x**2, 3: x**3 + x*y**2}

    assert decompose(x, True) == {x}
    assert decompose(x ** 2, True) == {x**2}
    assert decompose(x * y, True) == {x * y}
    assert decompose(x + y, True) == {x, y}
    assert decompose(x ** 2 + y, True) == {y, x ** 2}
    assert decompose(8 * x ** 2 + 4 * y + 7, True) == {7, 4*y, 8*x**2}
    assert decompose(x ** 2 + 3 * y * x, True) == {x ** 2, 3 * x * y}
    assert decompose(9 * x ** 2 + y + 4 * x + x ** 3 + y ** 2 * x + 3, True) == \
           {3, y, 4*x, 9*x**2, x*y**2, x**3}


def test_best_origin():
    expr1 = y ** 2 * x ** 5 + y ** 5 * x ** 7 + 7 * x + x ** 12 + y ** 7 * x

    l1 = Segment2D(Point(0, 3), Point(1, 1))
    l2 = Segment2D(Point(S(3) / 2, 0), Point(S(3) / 2, 3))
    l3 = Segment2D(Point(0, S(3) / 2), Point(3, S(3) / 2))
    l4 = Segment2D(Point(0, 2), Point(2, 0))
    l5 = Segment2D(Point(0, 2), Point(1, 1))
    l6 = Segment2D(Point(2, 0), Point(1, 1))

    assert best_origin((2, 1), 3, l1, expr1) == (0, 3)
    assert best_origin((2, 0), 3, l2, x ** 7) == (S(3) / 2, 0)
    assert best_origin((0, 2), 3, l3, x ** 7) == (0, S(3) / 2)
    assert best_origin((1, 1), 2, l4, x ** 7 * y ** 3) == (0, 2)
    assert best_origin((1, 1), 2, l4, x ** 3 * y ** 7) == (2, 0)
    assert best_origin((1, 1), 2, l5, x ** 2 * y ** 9) == (0, 2)
    assert best_origin((1, 1), 2, l6, x ** 9 * y ** 2) == (2, 0)


@slow
def test_polytope_integrate():
    #  Convex 2-Polytopes
    #  Vertex representation
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 2),
                                      Point(4, 0)), 1) == 4
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)), x * y) ==\
                                      Rational(1, 4)
    assert polytope_integrate(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)),
                              6*x**2 - 40*y) == Rational(-935, 3)

    assert polytope_integrate(Polygon(Point(0, 0), Point(0, sqrt(3)),
                                      Point(sqrt(3), sqrt(3)),
                                      Point(sqrt(3), 0)), 1) == 3

    hexagon = Polygon(Point(0, 0), Point(-sqrt(3) / 2, S.Half),
                      Point(-sqrt(3) / 2, S(3) / 2), Point(0, 2),
                      Point(sqrt(3) / 2, S(3) / 2), Point(sqrt(3) / 2, S.Half))

    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    #  Hyperplane representation
    assert polytope_integrate([((-1, 0), 0), ((1, 2), 4),
                               ((0, -1), 0)], 1) == 4
    assert polytope_integrate([((-1, 0), 0), ((0, 1), 1),
                               ((1, 0), 1), ((0, -1), 0)], x * y) == Rational(1, 4)
    assert polytope_integrate([((0, 1), 3), ((1, -2), -1),
                               ((-2, -1), -3)], 6*x**2 - 40*y) == Rational(-935, 3)
    assert polytope_integrate([((-1, 0), 0), ((0, sqrt(3)), 3),
                               ((sqrt(3), 0), 3), ((0, -1), 0)], 1) == 3

    hexagon = [((Rational(-1, 2), -sqrt(3) / 2), 0),
               ((-1, 0), sqrt(3) / 2),
               ((Rational(-1, 2), sqrt(3) / 2), sqrt(3)),
               ((S.Half, sqrt(3) / 2), sqrt(3)),
               ((1, 0), sqrt(3) / 2),
               ((S.Half, -sqrt(3) / 2), 0)]
    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    #  Non-convex polytopes
    #  Vertex representation
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(1, 1), Point(0, 0),
                                      Point(1, -1)), 1) == 3
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(0, 0), Point(1, 1),
                                      Point(1, -1), Point(0, 0)), 1) == 2
    #  Hyperplane representation
    assert polytope_integrate([((-1, 0), 1), ((0, 1), 1), ((1, -1), 0),
                               ((1, 1), 0), ((0, -1), 1)], 1) == 3
    assert polytope_integrate([((-1, 0), 1), ((1, 1), 0), ((-1, 1), 0),
                               ((1, 0), 1), ((-1, -1), 0),
                               ((1, -1), 0)], 1) == 2

    #  Tests for 2D polytopes mentioned in Chin et al(Page 10):
    #  http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
    fig1 = Polygon(Point(1.220, -0.827), Point(-1.490, -4.503),
                   Point(-3.766, -1.622), Point(-4.240, -0.091),
                   Point(-3.160, 4), Point(-0.981, 4.447),
                   Point(0.132, 4.027))
    assert polytope_integrate(fig1, x**2 + x*y + y**2) ==\
        S(2031627344735367)/(8*10**12)

    fig2 = Polygon(Point(4.561, 2.317), Point(1.491, -1.315),
                   Point(-3.310, -3.164), Point(-4.845, -3.110),
                   Point(-4.569, 1.867))
    assert polytope_integrate(fig2, x**2 + x*y + y**2) ==\
        S(517091313866043)/(16*10**11)

    fig3 = Polygon(Point(-2.740, -1.888), Point(-3.292, 4.233),
                   Point(-2.723, -0.697), Point(-0.643, -3.151))
    assert polytope_integrate(fig3, x**2 + x*y + y**2) ==\
        S(147449361647041)/(8*10**12)

    fig4 = Polygon(Point(0.211, -4.622), Point(-2.684, 3.851),
                   Point(0.468, 4.879), Point(4.630, -1.325),
                   Point(-0.411, -1.044))
    assert polytope_integrate(fig4, x**2 + x*y + y**2) ==\
        S(180742845225803)/(10**12)

    #  Tests for many polynomials with maximum degree given(2D case).
    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    polys = []
    expr1 = x**9*y + x**7*y**3 + 2*x**2*y**8
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    result_dict = polytope_integrate(tri, polys, max_degree=10)
    assert result_dict[expr1] == Rational(615780107, 594)
    assert result_dict[expr2] == Rational(13062161, 27)
    assert result_dict[expr3] == Rational(1946257153, 924)

    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    expr1 = x**7*y**1 + 2*x**2*y**6
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    assert polytope_integrate(tri, polys, max_degree=9) == \
        {x**7*y + 2*x**2*y**6: Rational(489262, 9)}

    #  Tests when all integral of all monomials up to a max_degree is to be
    #  calculated.
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)),
                              max_degree=4) == {0: 0, 1: 1, x: S.Half,
                                                x ** 2 * y ** 2: S.One / 9,
                                                x ** 4: S.One / 5,
                                                y ** 4: S.One / 5,
                                                y: S.Half,
                                                x * y ** 2: S.One / 6,
                                                y ** 2: S.One / 3,
                                                x ** 3: S.One / 4,
                                                x ** 2 * y: S.One / 6,
                                                x ** 3 * y: S.One / 8,
                                                x * y: S.One / 4,
                                                y ** 3: S.One / 4,
                                                x ** 2: S.One / 3,
                                                x * y ** 3: S.One / 8}

    #  Tests for 3D polytopes
    cube1 = [[(0, 0, 0), (0, 6, 6), (6, 6, 6), (3, 6, 0),
              (0, 6, 0), (6, 0, 6), (3, 0, 0), (0, 0, 6)],
             [1, 2, 3, 4], [3, 2, 5, 6], [1, 7, 5, 2], [0, 6, 5, 7],
             [1, 4, 0, 7], [0, 4, 3, 6]]
    assert polytope_integrate(cube1, 1) == S(162)

    #  3D Test cases in Chin et al(2015)
    cube2 = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],
             [3, 7, 6, 2], [1, 5, 7, 3], [5, 4, 6, 7], [0, 4, 5, 1],
             [2, 0, 1, 3], [2, 6, 4, 0]]

    cube3 = [[(0, 0, 0), (5, 0, 0), (5, 4, 0), (3, 2, 0), (3, 5, 0),
              (0, 5, 0), (0, 0, 5), (5, 0, 5), (5, 4, 5), (3, 2, 5),
              (3, 5, 5), (0, 5, 5)],
             [6, 11, 5, 0], [1, 7, 6, 0], [5, 4, 3, 2, 1, 0], [11, 10, 4, 5],
             [10, 9, 3, 4], [9, 8, 2, 3], [8, 7, 1, 2], [7, 8, 9, 10, 11, 6]]

    cube4 = [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (S.One / 4, S.One / 4, S.One / 4)],
             [0, 2, 1], [1, 3, 0], [4, 2, 3], [4, 3, 1],
             [0, 1, 2], [2, 4, 1], [0, 3, 2]]

    assert polytope_integrate(cube2, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           Rational(15625, 4)
    assert polytope_integrate(cube3, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(33835) / 12
    assert polytope_integrate(cube4, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(37) / 960

    #  Test cases from Mathematica's PolyhedronData library
    octahedron = [[(S.NegativeOne / sqrt(2), 0, 0), (0, S.One / sqrt(2), 0),
                   (0, 0, S.NegativeOne / sqrt(2)), (0, 0, S.One / sqrt(2)),
                   (0, S.NegativeOne / sqrt(2), 0), (S.One / sqrt(2), 0, 0)],
                  [3, 4, 5], [3, 5, 1], [3, 1, 0], [3, 0, 4], [4, 0, 2],
                  [4, 2, 5], [2, 0, 1], [5, 2, 1]]

    assert polytope_integrate(octahedron, 1) == sqrt(2) / 3

    great_stellated_dodecahedron =\
        [[(-0.32491969623290634095, 0, 0.42532540417601993887),
          (0.32491969623290634095, 0, -0.42532540417601993887),
          (-0.52573111211913359231, 0, 0.10040570794311363956),
          (0.52573111211913359231, 0, -0.10040570794311363956),
          (-0.10040570794311363956, -0.3090169943749474241, 0.42532540417601993887),
          (-0.10040570794311363956, 0.30901699437494742410, 0.42532540417601993887),
          (0.10040570794311363956, -0.3090169943749474241, -0.42532540417601993887),
          (0.10040570794311363956, 0.30901699437494742410, -0.42532540417601993887),
          (-0.16245984811645317047, -0.5, 0.10040570794311363956),
          (-0.16245984811645317047,  0.5, 0.10040570794311363956),
          (0.16245984811645317047,  -0.5, -0.10040570794311363956),
          (0.16245984811645317047,   0.5, -0.10040570794311363956),
          (-0.42532540417601993887, -0.3090169943749474241, -0.10040570794311363956),
          (-0.42532540417601993887, 0.30901699437494742410, -0.10040570794311363956),
          (-0.26286555605956679615, 0.1909830056250525759, -0.42532540417601993887),
          (-0.26286555605956679615, -0.1909830056250525759, -0.42532540417601993887),
          (0.26286555605956679615, 0.1909830056250525759, 0.42532540417601993887),
          (0.26286555605956679615, -0.1909830056250525759, 0.42532540417601993887),
          (0.42532540417601993887, -0.3090169943749474241, 0.10040570794311363956),
          (0.42532540417601993887, 0.30901699437494742410, 0.10040570794311363956)],
         [12, 3, 0, 6, 16], [17, 7, 0, 3, 13],
         [9, 6, 0, 7, 8], [18, 2, 1, 4, 14],
         [15, 5, 1, 2, 19], [11, 4, 1, 5, 10],
         [8, 19, 2, 18, 9], [10, 13, 3, 12, 11],
         [16, 14, 4, 11, 12], [13, 10, 5, 15, 17],
         [14, 16, 6, 9, 18], [19, 8, 7, 17, 15]]
    #  Actual volume is : 0.163118960624632
    assert Abs(polytope_integrate(great_stellated_dodecahedron, 1) -\
        0.163118960624632) < 1e-12

    expr = x **2 + y ** 2 + z ** 2
    octahedron_five_compound = [[(0, -0.7071067811865475244, 0),
                                 (0, 0.70710678118654752440, 0),
                                 (0.1148764602736805918,
                                  -0.35355339059327376220, -0.60150095500754567366),
                                 (0.1148764602736805918, 0.35355339059327376220,
                                     -0.60150095500754567366),
                                 (0.18587401723009224507,
                                  -0.57206140281768429760, 0.37174803446018449013),
                                 (0.18587401723009224507,  0.57206140281768429760,
                                  0.37174803446018449013),
                                 (0.30075047750377283683, -0.21850801222441053540,
                                  0.60150095500754567366),
                                 (0.30075047750377283683, 0.21850801222441053540,
                                  0.60150095500754567366),
                                 (0.48662449473386508189, -0.35355339059327376220,
                                  -0.37174803446018449013),
                                 (0.48662449473386508189, 0.35355339059327376220,
                                  -0.37174803446018449013),
                                 (-0.60150095500754567366, 0, -0.37174803446018449013),
                                 (-0.30075047750377283683, -0.21850801222441053540,
                                  -0.60150095500754567366),
                                 (-0.30075047750377283683, 0.21850801222441053540,
                                  -0.60150095500754567366),
                                 (0.60150095500754567366, 0, 0.37174803446018449013),
                                 (0.4156269377774534286, -0.57206140281768429760, 0),
                                 (0.4156269377774534286, 0.57206140281768429760, 0),
                                 (0.37174803446018449013, 0, -0.60150095500754567366),
                                 (-0.4156269377774534286, -0.57206140281768429760, 0),
                                 (-0.4156269377774534286, 0.57206140281768429760, 0),
                                 (-0.67249851196395732696, -0.21850801222441053540, 0),
                                 (-0.67249851196395732696, 0.21850801222441053540, 0),
                                 (0.67249851196395732696, -0.21850801222441053540, 0),
                                 (0.67249851196395732696, 0.21850801222441053540, 0),
                                 (-0.37174803446018449013, 0, 0.60150095500754567366),
                                 (-0.48662449473386508189, -0.35355339059327376220,
                                 0.37174803446018449013),
                                 (-0.48662449473386508189, 0.35355339059327376220,
                                  0.37174803446018449013),
                                 (-0.18587401723009224507, -0.57206140281768429760,
                                  -0.37174803446018449013),
                                 (-0.18587401723009224507, 0.57206140281768429760,
                                  -0.37174803446018449013),
                                 (-0.11487646027368059176, -0.35355339059327376220,
                                  0.60150095500754567366),
                                 (-0.11487646027368059176, 0.35355339059327376220,
                                 0.60150095500754567366)],
                                 [0, 10, 16], [23, 10, 0], [16, 13, 0],
                                 [0, 13, 23], [16, 10, 1], [1, 10, 23],
                                 [1, 13, 16], [23, 13, 1], [2, 4, 19],
                                 [22, 4, 2], [2, 19, 27], [27, 22, 2],
                                 [20, 5, 3], [3, 5, 21], [26, 20, 3],
                                 [3, 21, 26], [29, 19, 4], [4, 22, 29],
                                 [5, 20, 28], [28, 21, 5], [6, 8, 15],
                                 [17, 8, 6], [6, 15, 25], [25, 17, 6],
                                 [14, 9, 7], [7, 9, 18], [24, 14, 7],
                                 [7, 18, 24], [8, 12, 15], [17, 12, 8],
                                 [14, 11, 9], [9, 11, 18], [11, 14, 24],
                                 [24, 18, 11], [25, 15, 12], [12, 17, 25],
                                 [29, 27, 19], [20, 26, 28], [28, 26, 21],
                                 [22, 27, 29]]
    assert Abs(polytope_integrate(octahedron_five_compound, expr)) - 0.353553\
        < 1e-6

    cube_five_compound = [[(-0.1624598481164531631, -0.5, -0.6881909602355867691),
                           (-0.1624598481164531631, 0.5, -0.6881909602355867691),
                           (0.1624598481164531631, -0.5, 0.68819096023558676910),
                           (0.1624598481164531631, 0.5, 0.68819096023558676910),
                          (-0.52573111211913359231, 0, -0.6881909602355867691),
                          (0.52573111211913359231, 0, 0.68819096023558676910),
                          (-0.26286555605956679615, -0.8090169943749474241,
                           -0.1624598481164531631),
                          (-0.26286555605956679615, 0.8090169943749474241,
                           -0.1624598481164531631),
                          (0.26286555605956680301, -0.8090169943749474241,
                           0.1624598481164531631),
                          (0.26286555605956680301, 0.8090169943749474241,
                           0.1624598481164531631),
                          (-0.42532540417601993887, -0.3090169943749474241,
                           0.68819096023558676910),
                          (-0.42532540417601993887, 0.30901699437494742410,
                           0.68819096023558676910),
                          (0.42532540417601996609, -0.3090169943749474241,
                           -0.6881909602355867691),
                          (0.42532540417601996609, 0.30901699437494742410,
                           -0.6881909602355867691),
                          (-0.6881909602355867691, -0.5, 0.1624598481164531631),
                          (-0.6881909602355867691, 0.5,  0.1624598481164531631),
                          (0.68819096023558676910, -0.5, -0.1624598481164531631),
                          (0.68819096023558676910, 0.5, -0.1624598481164531631),
                          (-0.85065080835203998877, 0, -0.1624598481164531631),
                          (0.85065080835203993218, 0, 0.1624598481164531631)],
                          [18, 10, 3, 7], [13, 19, 8, 0], [18, 0, 8, 10],
                          [3, 19, 13, 7], [18, 7, 13, 0], [8, 19, 3, 10],
                          [6, 2, 11, 18], [1, 9, 19, 12], [11, 9, 1, 18],
                          [6, 12, 19, 2], [1, 12, 6, 18], [11, 2, 19, 9],
                          [4, 14, 11, 7], [17, 5, 8, 12], [4, 12, 8, 14],
                          [11, 5, 17, 7], [4, 7, 17, 12], [8, 5, 11, 14],
                          [6, 10, 15, 4], [13, 9, 5, 16], [15, 9, 13, 4],
                          [6, 16, 5, 10], [13, 16, 6, 4], [15, 10, 5, 9],
                          [14, 15, 1, 0], [16, 17, 3, 2], [14, 2, 3, 15],
                          [1, 17, 16, 0], [14, 0, 16, 2], [3, 17, 1, 15]]
    assert Abs(polytope_integrate(cube_five_compound, expr) - 1.25) < 1e-12

    echidnahedron = [[(0, 0, -2.4898982848827801995),
                      (0, 0, 2.4898982848827802734),
                      (0, -4.2360679774997896964, -2.4898982848827801995),
                      (0, -4.2360679774997896964, 2.4898982848827802734),
                      (0, 4.2360679774997896964, -2.4898982848827801995),
                      (0, 4.2360679774997896964, 2.4898982848827802734),
                      (-4.0287400534704067567, -1.3090169943749474241, -2.4898982848827801995),
                      (-4.0287400534704067567, -1.3090169943749474241, 2.4898982848827802734),
                      (-4.0287400534704067567, 1.3090169943749474241, -2.4898982848827801995),
                      (-4.0287400534704067567, 1.3090169943749474241, 2.4898982848827802734),
                      (4.0287400534704069747, -1.3090169943749474241, -2.4898982848827801995),
                      (4.0287400534704069747, -1.3090169943749474241, 2.4898982848827802734),
                      (4.0287400534704069747, 1.3090169943749474241, -2.4898982848827801995),
                      (4.0287400534704069747, 1.3090169943749474241, 2.4898982848827802734),
                      (-2.4898982848827801995, -3.4270509831248422723, -2.4898982848827801995),
                      (-2.4898982848827801995, -3.4270509831248422723, 2.4898982848827802734),
                      (-2.4898982848827801995, 3.4270509831248422723, -2.4898982848827801995),
                      (-2.4898982848827801995, 3.4270509831248422723, 2.4898982848827802734),
                      (2.4898982848827802734, -3.4270509831248422723, -2.4898982848827801995),
                      (2.4898982848827802734, -3.4270509831248422723, 2.4898982848827802734),
                      (2.4898982848827802734, 3.4270509831248422723, -2.4898982848827801995),
                      (2.4898982848827802734, 3.4270509831248422723, 2.4898982848827802734),
                      (-4.7169310137059934362, -0.8090169943749474241, -1.1135163644116066184),
                      (-4.7169310137059934362, 0.8090169943749474241, -1.1135163644116066184),
                      (4.7169310137059937438, -0.8090169943749474241, 1.11351636441160673519),
                      (4.7169310137059937438, 0.8090169943749474241, 1.11351636441160673519),
                      (-4.2916056095299737777, -2.1180339887498948482, 1.11351636441160673519),
                      (-4.2916056095299737777, 2.1180339887498948482, 1.11351636441160673519),
                      (4.2916056095299737777, -2.1180339887498948482, -1.1135163644116066184),
                      (4.2916056095299737777, 2.1180339887498948482, -1.1135163644116066184),
                      (-3.6034146492943870399, 0, -3.3405490932348205213),
                      (3.6034146492943870399, 0, 3.3405490932348202056),
                      (-3.3405490932348205213, -3.4270509831248422723, 1.11351636441160673519),
                      (-3.3405490932348205213, 3.4270509831248422723, 1.11351636441160673519),
                      (3.3405490932348202056, -3.4270509831248422723, -1.1135163644116066184),
                      (3.3405490932348202056, 3.4270509831248422723, -1.1135163644116066184),
                      (-2.9152236890588002395, -2.1180339887498948482, 3.3405490932348202056),
                      (-2.9152236890588002395, 2.1180339887498948482, 3.3405490932348202056),
                      (2.9152236890588002395, -2.1180339887498948482, -3.3405490932348205213),
                      (2.9152236890588002395, 2.1180339887498948482, -3.3405490932348205213),
                      (-2.2270327288232132368, 0, -1.1135163644116066184),
                      (-2.2270327288232132368, -4.2360679774997896964, -1.1135163644116066184),
                      (-2.2270327288232132368, 4.2360679774997896964, -1.1135163644116066184),
                      (2.2270327288232134704, 0, 1.11351636441160673519),
                      (2.2270327288232134704, -4.2360679774997896964, 1.11351636441160673519),
                      (2.2270327288232134704, 4.2360679774997896964, 1.11351636441160673519),
                      (-1.8017073246471935200, -1.3090169943749474241, 1.11351636441160673519),
                      (-1.8017073246471935200, 1.3090169943749474241, 1.11351636441160673519),
                      (1.8017073246471935043, -1.3090169943749474241, -1.1135163644116066184),
                      (1.8017073246471935043, 1.3090169943749474241, -1.1135163644116066184),
                      (-1.3763819204711735382, 0, -4.7169310137059934362),
                      (-1.3763819204711735382, 0, 0.26286555605956679615),
                      (1.37638192047117353821, 0, 4.7169310137059937438),
                      (1.37638192047117353821, 0, -0.26286555605956679615),
                      (-1.1135163644116066184, -3.4270509831248422723, -3.3405490932348205213),
                      (-1.1135163644116066184, -0.8090169943749474241, 4.7169310137059937438),
                      (-1.1135163644116066184, -0.8090169943749474241, -0.26286555605956679615),
                      (-1.1135163644116066184, 0.8090169943749474241, 4.7169310137059937438),
                      (-1.1135163644116066184, 0.8090169943749474241, -0.26286555605956679615),
                      (-1.1135163644116066184, 3.4270509831248422723, -3.3405490932348205213),
                      (1.11351636441160673519, -3.4270509831248422723, 3.3405490932348202056),
                      (1.11351636441160673519, -0.8090169943749474241, -4.7169310137059934362),
                      (1.11351636441160673519, -0.8090169943749474241, 0.26286555605956679615),
                      (1.11351636441160673519, 0.8090169943749474241, -4.7169310137059934362),
                      (1.11351636441160673519, 0.8090169943749474241, 0.26286555605956679615),
                      (1.11351636441160673519, 3.4270509831248422723, 3.3405490932348202056),
                      (-0.85065080835203998877, 0, 1.11351636441160673519),
                      (0.85065080835203993218, 0, -1.1135163644116066184),
                      (-0.6881909602355867691, -0.5, -1.1135163644116066184),
                      (-0.6881909602355867691, 0.5, -1.1135163644116066184),
                      (-0.6881909602355867691, -4.7360679774997896964, -1.1135163644116066184),
                      (-0.6881909602355867691, -2.1180339887498948482, -1.1135163644116066184),
                      (-0.6881909602355867691, 2.1180339887498948482, -1.1135163644116066184),
                      (-0.6881909602355867691, 4.7360679774997896964, -1.1135163644116066184),
                      (0.68819096023558676910, -0.5, 1.11351636441160673519),
                      (0.68819096023558676910, 0.5, 1.11351636441160673519),
                      (0.68819096023558676910, -4.7360679774997896964, 1.11351636441160673519),
                      (0.68819096023558676910, -2.1180339887498948482, 1.11351636441160673519),
                      (0.68819096023558676910, 2.1180339887498948482, 1.11351636441160673519),
                      (0.68819096023558676910, 4.7360679774997896964, 1.11351636441160673519),
                      (-0.42532540417601993887, -1.3090169943749474241, -4.7169310137059934362),
                      (-0.42532540417601993887, -1.3090169943749474241, 0.26286555605956679615),
                      (-0.42532540417601993887, 1.3090169943749474241, -4.7169310137059934362),
                      (-0.42532540417601993887, 1.3090169943749474241, 0.26286555605956679615),
                      (-0.26286555605956679615, -0.8090169943749474241, 1.11351636441160673519),
                      (-0.26286555605956679615, 0.8090169943749474241, 1.11351636441160673519),
                      (0.26286555605956679615, -0.8090169943749474241, -1.1135163644116066184),
                      (0.26286555605956679615, 0.8090169943749474241, -1.1135163644116066184),
                      (0.42532540417601996609, -1.3090169943749474241, 4.7169310137059937438),
                      (0.42532540417601996609, -1.3090169943749474241, -0.26286555605956679615),
                      (0.42532540417601996609, 1.3090169943749474241, 4.7169310137059937438),
                      (0.42532540417601996609, 1.3090169943749474241, -0.26286555605956679615)],
                      [9, 66, 47], [44, 62, 77], [20, 91, 49], [33, 47, 83],
                      [3, 77, 84], [12, 49, 53], [36, 84, 66], [28, 53, 62],
                      [73, 83, 91], [15, 84, 46], [25, 64, 43], [16, 58, 72],
                      [26, 46, 51], [11, 43, 74], [4, 72, 91], [60, 74, 84],
                      [35, 91, 64], [23, 51, 58], [19, 74, 77], [79, 83, 78],
                      [6, 56, 40], [76, 77, 81], [21, 78, 75], [8, 40, 58],
                      [31, 75, 74], [42, 58, 83], [41, 81, 56], [13, 75, 43],
                      [27, 51, 47], [2, 89, 71], [24, 43, 62], [17, 47, 85],
                      [14, 71, 56], [65, 85, 75], [22, 56, 51], [34, 62, 89],
                      [5, 85, 78], [32, 81, 46], [10, 53, 48], [45, 78, 64],
                      [7, 46, 66], [18, 48, 89], [37, 66, 85], [70, 89, 81],
                      [29, 64, 53], [88, 74, 1], [38, 67, 48], [42, 83, 72],
                      [57, 1, 85], [34, 48, 62], [59, 72, 87], [19, 62, 74],
                      [63, 87, 67], [17, 85, 83], [52, 75, 1], [39, 87, 49],
                      [22, 51, 40], [55, 1, 66], [29, 49, 64], [30, 40, 69],
                      [13, 64, 75], [82, 69, 87], [7, 66, 51], [90, 85, 1],
                      [59, 69, 72], [70, 81, 71], [88, 1, 84], [73, 72, 83],
                      [54, 71, 68], [5, 83, 85], [50, 68, 69], [3, 84, 81],
                      [57, 66, 1], [30, 68, 40], [28, 62, 48], [52, 1, 74],
                      [23, 40, 51], [38, 48, 86], [9, 51, 66], [80, 86, 68],
                      [11, 74, 62], [55, 84, 1], [54, 86, 71], [35, 64, 49],
                      [90, 1, 75], [41, 71, 81], [39, 49, 67], [15, 81, 84],
                      [61, 67, 86], [21, 75, 64], [24, 53, 43], [50, 69, 0],
                      [37, 85, 47], [31, 43, 75], [61, 0, 67], [27, 47, 58],
                      [10, 67, 53], [8, 58, 69], [90, 75, 85], [45, 91, 78],
                      [80, 68, 0], [36, 66, 46], [65, 78, 85], [63, 0, 87],
                      [32, 46, 56], [20, 87, 91], [14, 56, 68], [57, 85, 66],
                      [33, 58, 47], [61, 86, 0], [60, 84, 77], [37, 47, 66],
                      [82, 0, 69], [44, 77, 89], [16, 69, 58], [18, 89, 86],
                      [55, 66, 84], [26, 56, 46], [63, 67, 0], [31, 74, 43],
                      [36, 46, 84], [50, 0, 68], [25, 43, 53], [6, 68, 56],
                      [12, 53, 67], [88, 84, 74], [76, 89, 77], [82, 87, 0],
                      [65, 75, 78], [60, 77, 74], [80, 0, 86], [79, 78, 91],
                      [2, 86, 89], [4, 91, 87], [52, 74, 75], [21, 64, 78],
                      [18, 86, 48], [23, 58, 40], [5, 78, 83], [28, 48, 53],
                      [6, 40, 68], [25, 53, 64], [54, 68, 86], [33, 83, 58],
                      [17, 83, 47], [12, 67, 49], [41, 56, 71], [9, 47, 51],
                      [35, 49, 91], [2, 71, 86], [79, 91, 83], [38, 86, 67],
                      [26, 51, 56], [7, 51, 46], [4, 87, 72], [34, 89, 48],
                      [15, 46, 81], [42, 72, 58], [10, 48, 67], [27, 58, 51],
                      [39, 67, 87], [76, 81, 89], [3, 81, 77], [8, 69, 40],
                      [29, 53, 49], [19, 77, 62], [22, 40, 56], [20, 49, 87],
                      [32, 56, 81], [59, 87, 69], [24, 62, 53], [11, 62, 43],
                      [14, 68, 71], [73, 91, 72], [13, 43, 64], [70, 71, 89],
                      [16, 72, 69], [44, 89, 62], [30, 69, 68], [45, 64, 91]]
    #  Actual volume is : 51.405764746872634
    assert Abs(polytope_integrate(echidnahedron, 1) - 51.4057647468726) < 1e-12
    assert Abs(polytope_integrate(echidnahedron, expr) - 253.569603474519) <\
    1e-12

    #  Tests for many polynomials with maximum degree given(2D case).
    assert polytope_integrate(cube2, [x**2, y*z], max_degree=2) == \
        {y * z: 3125 / S(4), x ** 2: 3125 / S(3)}

    assert polytope_integrate(cube2, max_degree=2) == \
        {1: 125, x: 625 / S(2), x * z: 3125 / S(4), y: 625 / S(2),
         y * z: 3125 / S(4), z ** 2: 3125 / S(3), y ** 2: 3125 / S(3),
         z: 625 / S(2), x * y: 3125 / S(4), x ** 2: 3125 / S(3)}

def test_point_sort():
    assert point_sort([Point(0, 0), Point(1, 0), Point(1, 1)]) == \
        [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]

    fig6 = Polygon((0, 0), (1, 0), (1, 1))
    assert polytope_integrate(fig6, x*y) == Rational(-1, 8)
    assert polytope_integrate(fig6, x*y, clockwise = True) == Rational(1, 8)


def test_polytopes_intersecting_sides():
    fig5 = Polygon(Point(-4.165, -0.832), Point(-3.668, 1.568),
                   Point(-3.266, 1.279), Point(-1.090, -2.080),
                   Point(3.313, -0.683), Point(3.033, -4.845),
                   Point(-4.395, 4.840), Point(-1.007, -3.328))
    assert polytope_integrate(fig5, x**2 + x*y + y**2) ==\
        S(1633405224899363)/(24*10**12)

    fig6 = Polygon(Point(-3.018, -4.473), Point(-0.103, 2.378),
                   Point(-1.605, -2.308), Point(4.516, -0.771),
                   Point(4.203, 0.478))
    assert polytope_integrate(fig6, x**2 + x*y + y**2) ==\
        S(88161333955921)/(3*10**12)


def test_max_degree():
    polygon = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    polys = [1, x, y, x*y, x**2*y, x*y**2]
    assert polytope_integrate(polygon, polys, max_degree=3) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    assert polytope_integrate(polygon, polys, max_degree=2) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4)}
    assert polytope_integrate(polygon, polys, max_degree=1) == \
        {1: 1, x: S.Half, y: S.Half}


def test_main_integrate3d():
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    vertices = cube[0]
    faces = cube[1:]
    hp_params = hyperplane_parameters(faces, vertices)
    assert main_integrate3d(1, faces, vertices, hp_params) == -125
    assert main_integrate3d(1, faces, vertices, hp_params, max_degree=1) == \
        {1: -125, y: Rational(-625, 2), z: Rational(-625, 2), x: Rational(-625, 2)}


def test_main_integrate():
    triangle = Polygon((0, 3), (5, 3), (1, 1))
    facets = triangle.sides
    hp_params = hyperplane_parameters(triangle)
    assert main_integrate(x**2 + y**2, facets, hp_params) == Rational(325, 6)
    assert main_integrate(x**2 + y**2, facets, hp_params, max_degree=1) == \
        {0: 0, 1: 5, y: Rational(35, 3), x: 10}


def test_polygon_integrate():
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    facet = cube[1]
    facets = cube[1:]
    vertices = cube[0]
    assert polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0) == -25


def test_distance_to_side():
    point = (0, 0, 0)
    assert distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0)) == -sqrt(2)/2


def test_lineseg_integrate():
    polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    line_seg = [(0, 5, 0), (5, 5, 0)]
    assert lineseg_integrate(polygon, 0, line_seg, 1, 0) == 5
    assert lineseg_integrate(polygon, 0, line_seg, 0, 0) == 0


def test_integration_reduction():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    assert integration_reduction(facets, 0, a, b, 1, (x, y), 0) == 5
    assert integration_reduction(facets, 0, a, b, 0, (x, y), 0) == 0


def test_integration_reduction_dynamic():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    x0 = facets[0].points[0]
    monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                       [y, 0, 1, 15], [x, 1, 0, None]]

    assert integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == Rational(25, 2)
    assert integration_reduction_dynamic(facets, 0, a, b, 0, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == 0


def test_is_vertex():
    assert is_vertex(2) is False
    assert is_vertex((2, 3)) is True
    assert is_vertex(Point(2, 3)) is True
    assert is_vertex((2, 3, 4)) is True
    assert is_vertex((2, 3, 4, 5)) is False


def test_issue_19234():
    polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    polys =  [ 1, x, y, x*y, x**2*y, x*y**2]
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    polys =  [ 1, x, y, x*y, 3 + x**2*y, x + x*y**2]
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y + 3: Rational(19, 6), x*y**2 + x: Rational(2, 3)}
