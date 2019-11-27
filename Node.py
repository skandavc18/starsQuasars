import sys
import time
import numpy as np

class Node(object):
    def __init__(self):
        self.right = None
        self.split_val = None
        self.W = None
        self.leaf = False
        self.left = None
        self.split_id = None

    def pred(self, x):
        if self.leaf == False:
            if x[self.split_id] <= self.split_val:
                return self.left.pred(x)
            else:
                return self.right.pred(x)
        else:
            return self.W

    def LW(self, grad, hns, lambd):
        a = np.sum(grad)
        b = np.sum(hns)
        c = (b + lambd)
        return a / c

    def build(self, inst, grad, hns, rate, depth, parameter):
        assert inst.shape[0] == len(grad) == len(hns)
        if depth > parameter['dep']:
            self.leaf = True
            self.W = self.LW(grad, hns, parameter['lambda']) * rate
            return
        
        bg = 0.
        bli = None
        bfi = None
        G = np.sum(grad)
        H = np.sum(hns)
        bv = 0.
        bri = None
        for fi in range(inst.shape[1]):
            G_l, H_l = 0., 0.
            sii = inst[:,fi].argsort()
            for j in range(sii.shape[0]):
                H_l += hns[sii[j]]
                G_l += grad[sii[j]]
                H_r = H - H_l
                G_r = G - G_l
                cg = self.split_gain(G, H, G_l, H_l, G_r, H_r, parameter['lambda'])
                if bg < cg:
                    bfi = fi
                    bg = cg
                    bv = inst[sii[j]][fi]
                    bli = sii[:j+1]
                    bri = sii[j+1:]
        if bg >= parameter['msg']:
            self.split_id = bfi
            self.split_val = bv

            self.left = Node()
            self.left.build(inst[bli],grad[bli],hns[bli],rate,depth+1, parameter)

            self.right = Node()
            self.right.build(inst[bri],grad[bri],hns[bri],rate,depth+1, parameter)            
        else:
            self.leaf = True
            self.W = self.LW(grad, hns, parameter['lambda']) * rate
    
    def split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        def term(g, h):
            zx = np.square(g)
            py = (h + lambd)
            return zx / py
        return  term(G_r, H_r) - term(G, H) + term(G_l, H_l)