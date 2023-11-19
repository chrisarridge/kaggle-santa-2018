"""Test against optimal costs on TSPLIB95 website
"""

import unittest
import santaslittletsplib
import santaslittletsplib.metrics

class TestLibProblems(unittest.TestCase):
    def test_tsp_a280_euc2d(self):
        tour = santaslittletsplib.load('data/tsplib95/tsp/a280.opt.tour.gz')
        problem = santaslittletsplib.load('data/tsplib95/tsp/a280.tsp.gz')
        self.assertEqual(problem.cost(tour), 2579)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.Euclidean()), 2579)

    def test_tsp_pcb442_euc2d(self):
        tour = santaslittletsplib.load('data/tsplib95/tsp/pcb442.opt.tour.gz')
        problem = santaslittletsplib.load('data/tsplib95/tsp/pcb442.tsp.gz')
        self.assertEqual(problem.cost(tour), 50778)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.Euclidean()), 50778)

    def test_tsp_gr9882_euc2d(self):
        tour = santaslittletsplib.load('data/othervalidation/gr9882.tour')
        problem = santaslittletsplib.load('data/othervalidation/gr9882.tsp')
        self.assertEqual(problem.cost(tour), 300899)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.Euclidean()), 300899)

    def test_tsp_xql662_euc2d(self):
        tour = santaslittletsplib.load('data/othervalidation/xql662.tour')
        problem = santaslittletsplib.load('data/othervalidation/xql662.tsp')
        self.assertEqual(problem.cost(tour), 2513)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.Euclidean()), 2513)

    def test_tsp_att48_att(self):
        tour = santaslittletsplib.load('data/tsplib95/tsp/att48.opt.tour.gz')
        problem = santaslittletsplib.load('data/tsplib95/tsp/att48.tsp.gz')
        self.assertEqual(problem.cost(tour), 10628)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.AttPseudoEuclidean()), 10628)

    def test_tsp_gr202_geo(self):
        tour = santaslittletsplib.load('data/tsplib95/tsp/gr202.opt.tour.gz')
        problem = santaslittletsplib.load('data/tsplib95/tsp/gr202.tsp.gz')
        metric = santaslittletsplib.metrics.Geographical()
        self.assertEqual(metric([0,1], problem._nodes), 1449)
        self.assertEqual(metric([0,2], problem._nodes), 1514)
        self.assertEqual(metric([0,3], problem._nodes), 1735)
        self.assertEqual(metric([0,4], problem._nodes), 1721)
        self.assertEqual(metric([0,5], problem._nodes), 1884)
        self.assertEqual(metric([0,6], problem._nodes), 1939)
        self.assertEqual(problem.cost(tour), 40160)
        self.assertEqual(problem.cost(tour, metric), 40160)

    def test_tsp_ulysses16_geo(self):
        tour = santaslittletsplib.load('data/tsplib95/tsp/ulysses16.opt.tour.gz')
        problem = santaslittletsplib.load('data/tsplib95/tsp/ulysses16.tsp.gz')
        self.assertEqual(problem.cost(tour), 6859)
        self.assertEqual(problem.cost(tour, santaslittletsplib.metrics.Geographical()), 6859)

    def test_tsp_bays29_explicit_fullmatrix(self):
        problem = santaslittletsplib.load('data/tsplib95/tsp/bays29.tsp.gz')
        tour = santaslittletsplib.load('data/tsplib95/tsp/bays29.opt.tour.gz')
        self.assertEqual(problem.cost(tour), 2020)
        self.assertEqual(problem.cost(tour, None), 2020)

    def test_tsp_bayg29_explicit_upperrow(self):
        problem = santaslittletsplib.load('data/tsplib95/tsp/bayg29.tsp.gz')
        tour = santaslittletsplib.load('data/tsplib95/tsp/bayg29.opt.tour.gz')
        self.assertEqual(problem.cost(tour), 1610)
        self.assertEqual(problem.cost(tour, None), 1610)

    def test_tsp_fri26_explicit_lowerdiag(self):
        problem = santaslittletsplib.load('data/tsplib95/tsp/fri26.tsp.gz')
        tour = santaslittletsplib.load('data/tsplib95/tsp/fri26.opt.tour.gz')
        self.assertEqual(problem.cost(tour), 937)
        self.assertEqual(problem.cost(tour, None), 937)

if __name__=="__main__":
    unittest.main()