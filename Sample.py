__author__ = 'Daguerreo'

import numpy as np

class Sample:
    # __ private
    def __init__(self):
        self.path = ''
        self.hist2d = np.array([])
        self.label = ''
        self.sift = []
