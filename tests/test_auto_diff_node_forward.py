import unittest
from parameterized import parameterized
from gradescope_utils.autograder_utils.decorators import weight

from typing import List
import numpy as np
import auto_diff as ad

class TestAutoDiffNodeForward(unittest.TestCase):
    def check_compute_output(self,
        node: ad.Node, input_values: List[np.ndarray], expected_output: np.ndarray
    ) -> None:
        output = node.op.compute(node, input_values)
        np.testing.assert_allclose(actual=output, desired=expected_output)

    def test_mul(self):
        x1 = ad.Variable("x1")
        x2 = ad.Variable("x2")
        y = ad.mul(x1, x2)

        self.check_compute_output(
            y,
            [
                np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
                np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            ],
            np.array([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]]),
        )

    def test_mul_by_const(self):
        x1 = ad.Variable("x1")
        y = ad.mul_by_const(x1, 2.7)

        self.check_compute_output(
            y,
            [np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
            np.array([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]]),
        )

    def test_div(self):
        x1 = ad.Variable("x1")
        x2 = ad.Variable("x2")
        y = ad.div(x1, x2)

        self.check_compute_output(
            y,
            [
                np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
                np.array([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
            ],
            np.array([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        )

    def test_div_by_const(self):
        x1 = ad.Variable("x1")
        y = ad.div_by_const(x1, 5.0)

        self.check_compute_output(
            y,
            [np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
            np.array([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]]),
        )

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_matmul(self, trans_A, trans_B):
        x1 = ad.Variable("x1")
        x2 = ad.Variable("x2")
        y = ad.matmul(x1, x2, trans_A=trans_A, trans_B=trans_B)

        x1_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x2_val = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        if trans_A:
            x1_val = x1_val.T
        if trans_B:
            x2_val = x2_val.T

        self.check_compute_output(
            y,
            [x1_val, x2_val],
            np.array([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]),
        )
