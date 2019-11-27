import sys
import time
import numpy as np
from Node import *

class Tree(object):
    def __init__(self):
        self.root = None

    def pred(self, x):
        return self.root.pred(x)

    def build(self, inst, grad, hns, rate, parameter):
        assert len(inst) == len(grad) == len(hns)
        cdr = 0
        self.root = Node()
        self.root.build(inst, grad, hns, rate, cdr, parameter)    