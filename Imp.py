import sys
import time
NUMBER = sys.maxsize
import numpy as np
from Tree import *

class Data(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

class Gradient_Boosted_Tree(object):
    def __init__(self):
        self.bst = None
        self.parameters = {'gamma': 0.,'lambda': 1.,'msg': 0.1,'dep': 5,'learning_rate': 0.22,}

    def pred(self, x, models=None, num_iteration=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.pred(x) for m in models[:num_iteration])

    def gradient(self, ts, scores):
        lab = ts.y
        hns = np.full(len(lab), 2)
        if scores is not None:
            grad = np.array([2 * (lab[i] - scores[i]) for i in range(len(lab))])          
        else:
            grad = np.random.uniform(size=len(lab))
        return grad, hns

    def DST(self, ts, models):
        if len(models) == 0:
            return None
        X = ts.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.pred(X[i], models=models)
        return scores

    def loss(self, models, ds):
        errors = []
        for x, y in zip(ds.X, ds.y):
            errors.append(y - self.pred(x, models))
        return np.mean(np.square(errors))

    def build(self, ts, grad, hns, rate):
        learner = Tree()
        learner.build(ts.X, grad, hns, rate, self.parameters)
        return learner
    
    def GM(self, ts, scores):
        return self.gradient(ts, scores)
    
    def LM(self, models, ds):
        return self.loss(models, ds)

    def train(self, parameters, ts, boosts=20, vss=None, esr=5):
        self.parameters.update(parameters)
        models = []
        rate = 1.
        bst = None
        bv_loss = NUMBER
        test = time.time()

        print("Training until validation scores don't improve for {} rounds.".format(esr))
        for iteration in range(boosts):
            ist = time.time()
            scores = self.DST(ts, models)
            grad, hns = self.GM(ts, scores)
            learner = self.build(ts, grad, hns, rate)
            if iteration > 0:
                rate *= self.parameters['learning_rate']
            models.append(learner)
            tll = self.LM(models, ts)
            vll = self.LM(models, vss) if vss else None
            vlls = '{:.10f}'.format(vll) if vll else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iteration, tll, vlls, time.time() - ist))
            if vll is not None and vll < bv_loss:
                bv_loss = vll
                bst = iteration
            if iteration - bst >= esr:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(bst, bv_loss))
                break

        self.models = models
        self.bst = bst
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - test))
    


