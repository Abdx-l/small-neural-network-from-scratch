import unittest
import numpy as np
from network import Network, gradient_error


class NetworkTests(unittest.TestCase):
    def test_finite_difference_gradients(self):
        model = Network([2,3,2], seed=5)
        x = np.array([[0.2,0.8],[0.9,0.3],[0.4,0.6]])
        labels = np.array([0,1,0])
        self.assertLess(gradient_error(model,x,labels),1e-6)

    def test_training_improves_non_linear_problem(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(400,2))
        y = (x[:,0]*x[:,1] > 0).astype(np.int64)
        model = Network([2,16,2], seed=5)
        history = model.train(x,y,epochs=100,batch_size=64)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["accuracy"], .9)


if __name__ == "__main__": unittest.main()

