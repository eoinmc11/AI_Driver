import numpy as np


class SumTree(object):
    data_pointer = 0  # Which leaf are we are (L to R) | Index in data

    def __init__(self, capacity):
        self.capacity = capacity  # Trees capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add_experience(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1  # What index to put priority
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_index, priority)  # update tree_frame

        # Go back to the start if full
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]  # Change = new priority score - former priority score
        self.tree[tree_index] = priority  # Set new priority
        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop
            tree_index = (tree_index - 1) // 2  # Accesses the parent leaf
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        SEARCH FOR PRIORITY AND EXPERIENCE

        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
           1   2
          / \ / \
         3  4 5  6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1  # this leaf's left and right kids
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):  # reach bottom, end search
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # the root
