import numpy as np

from v3_PER_D3QN.ReplayMemory import SumTree as St


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    per_epsilon = 0.01  # small amount to avoid zero priority
    per_alpha = 0.6  # [0~1] convert the importance of TD error to priority
    per_beta = 0.4  # importance-sampling, from initial value increasing to 1
    per_beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = St.SumTree(capacity)

    def store(self, transition):
        # Find Max Priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add_experience(max_priority, transition)  # set the max p for new p

    def sample(self, n):
        minibatch = []
        batch_index = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            batch_index[i] = index
            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return batch_index, minibatch

    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.per_epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.per_alpha)
        for ti, p in zip(tree_index, ps):
            self.tree.update(ti, p)
