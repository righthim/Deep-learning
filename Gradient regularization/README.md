The gradient regularization is a regularization method that adds the loss gradient norm penalty to the loss:

$\widetilde{L}(\theta):=L(\theta)+\lambda\lVert \nabla L(\theta)\rVert^2$.

The goal is to arrive at the 'flat minima' of the loss, i.e. among many local minima, we want to find the minima that have small gradients near the minima. It is considered that the sharp minima is data specific, and flat minima is better generalize in the literature. Although the importance of the flat minima in generalization has long been acknowledge([Flat minima](https://www.bioinf.jku.at/publications/older/3304.pdf)), the idea of adding gradient norm penalty seems to be a relatively recent suggestion[(Implicit graidnet regularization)](https://arxiv.org/abs/2009.11162), [(Explicit gradient regularization)](https://arxiv.org/abs/2202.03599).

There are two approaches in gradient regularization: Implict and Explicit. The former usually denotes the stochastic gradient descent, which is known to minimize the gradient norm penalized loss. The latter adds the gradient norm penalty explicitly, usually by finite difference or other numerical methods. Some paper argues finite difference method is efficient and also have additional generalization effect([Gradient regularization via FDM](https://arxiv.org/abs/2210.02720)). However, in my code the penalty uses the auto differentiation of the pytorch, which is not finite difference method.



<p align="center">
  <img src="./experiment results/gradient penalty lamb 1e-05.jpg",align='center',width='49%'>
  <img src="./experiment results/gradient penalty lamb 1.jpg",align='center',width='49%'>
  MNIST classification comparison. MLP model with tanh activation, 1-layer with 512 hidden nodes, (Left) $\lambda=10^{-5}$ (Right) $\lambda=1$. 1% improved.
</p>
