import sys
import time
NUMBER = sys.maxsize
import numpy as np


class Data(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

class GBT(object):
    def __init__(self):
        self.parameters = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.1,
                       'max_depth': 5,
                       'learning_rate': 0.22,
                       }
        self.best_iteration = None

    def predict(self, x, models=None, num_iteration=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.predict(x) for m in models[:num_iteration])

    def gradient(self, train_set, scores):
        labels = train_set.y
        hessian = np.full(len(labels), 2)
        if scores is not None:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])          
        else:
            grad = np.random.uniform(size=len(labels))
        return grad, hessian

    def data_scores(self, train_set, models):
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores

    def loss(self, models, data_set):
        errors = []
        for x, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.mean(np.square(errors))

    def build(self, train_set, grad, hessian, shrinkage_rate):
        learner = Tree()
        learner.build(train_set.X, grad, hessian, shrinkage_rate, self.parameters)
        return learner
    
    def gradient_main(self, train_set, scores):
        return self.gradient(train_set, scores)
    
    def loss_main(self, models, data_set):
        return self.loss(models, data_set)

    def train(self, parameters, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
        self.parameters.update(parameters)
        models = []
        shrinkage_rate = 1.
        best_iteration = None
        bv_loss = NUMBER
        train_start_time = time.time()

        print("Training until validation scores don't improve for {} rounds."
              .format(early_stopping_rounds))
        for iteration in range(num_boost_round):
            iter_start_time = time.time()
            scores = self.data_scores(train_set, models)
            grad, hessian = self.gradient_main(train_set, scores)
            learner = self.build(train_set, grad, hessian, shrinkage_rate)
            if iteration > 0:
                shrinkage_rate *= self.parameters['learning_rate']
            models.append(learner)
            train_loss = self.loss_main(models, train_set)
            val_loss = self.loss_main(models, valid_set) if valid_set else None
            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iteration, train_loss, val_loss_str, time.time() - iter_start_time))
            if val_loss is not None and val_loss < bv_loss:
                bv_loss = val_loss
                best_iteration = iteration
            if iteration - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(best_iteration, bv_loss))
                break

        self.models = models
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

    
class Tree(object):
    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, shrinkage_rate, parameter):
        assert len(instances) == len(grad) == len(hessian)
        self.root = Node()
        current_depth = 0
        self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, parameter)

    def predict(self, x):
        return self.root.predict(x)

class Node(object):
    def __init__(self):
        self.right = None
        self.split_val = None
        self.W = None
        self.leaf = False
        self.left = None
        self.split_id = None

    def predict(self, x):
        if self.leaf:
            return self.W
        else:
            if x[self.split_id] <= self.split_val:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    def leaf_weight(self, grad, hessian, lambd):
        return np.sum(grad) / (np.sum(hessian) + lambd)

    def build(self, instances, grad, hessian, shrinkage_rate, depth, parameter):
        assert instances.shape[0] == len(grad) == len(hessian)
        if depth > parameter['max_depth']:
            self.leaf = True
            self.W = self.leaf_weight(grad, hessian, parameter['lambda']) * shrinkage_rate
            return
        G = np.sum(grad)
        H = np.sum(hessian)
        bg = 0.
        bfi = None
        bv = 0.
        bli = None
        bri = None
        for fi in range(instances.shape[1]):
            G_l, H_l = 0., 0.
            sii = instances[:,fi].argsort()
            for j in range(sii.shape[0]):
                G_l += grad[sii[j]]
                H_l += hessian[sii[j]]
                G_r = G - G_l
                H_r = H - H_l
                cg = self.split_gain(G, H, G_l, H_l, G_r, H_r, parameter['lambda'])
                if cg > bg:
                    bg = cg
                    bfi = fi
                    bv = instances[sii[j]][fi]
                    bli = sii[:j+1]
                    bri = sii[j+1:]
        if bg >= parameter['min_split_gain']:
            self.split_id = bfi
            self.split_val = bv

            self.left = Node()
            self.left.build(instances[bli],grad[bli],hessian[bli],shrinkage_rate,depth+1, parameter)

            self.right = Node()
            self.right.build(instances[bri],grad[bri],hessian[bri],shrinkage_rate,depth+1, parameter)            
        else:
            self.leaf = True
            self.W = self.leaf_weight(grad, hessian, parameter['lambda']) * shrinkage_rate
    
    def split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        def term(g, h):
            return np.square(g) / (h + lambd)
        return term(G_l, H_l) + term(G_r, H_r) - term(G, H)