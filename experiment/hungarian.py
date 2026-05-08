import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianAssigner:
    def __init__(self, distance_matrix, score_matrix, step, step_change, weight, threshold):
        self.threshold = threshold
        self.distance_matrix = distance_matrix
        self.score_matrix = score_matrix
        self.step = step
        self.step_change = step_change
        self.weight = weight

    def should_assign(self):
        benefit_matrix = self.weight * self.score_matrix + (1 - self.weight) * self.distance_matrix
        # 兜底策略
        cond_step = (self.step % self.step_change == 0)
        cond_threshold = (benefit_matrix.max() >= self.threshold)
        return cond_step or cond_threshold

    def assign(self):
        m, n = self.distance_matrix.shape
        benefit_matrix = self.weight * self.score_matrix + (1 - self.weight) * self.distance_matrix
        benefit_matrix = np.nan_to_num(benefit_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        cost_matrix = -benefit_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_vector = -np.ones(m, dtype=int)
        assignment_vector[row_ind] = col_ind
        return assignment_vector
