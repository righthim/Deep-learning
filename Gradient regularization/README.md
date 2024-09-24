The gradient regularization is a regularization method that adds the loss gradient norm penalty to the loss:

$\widetilde{L}(\theta):=L(\theta)+\lambda\lVert \nabla L(\theta)\rVert^2$.

The goal is to arrive at the 'flat minima' of the loss, i.e. among many local minima, we want to find the minima that have small gradients near the minima. It is considered that the sharp minima is data specific, and flat minima is better generalize in the literature. Although the importance of the flat minima in generalization has long been acknowledge([Flat minima](https://www.bioinf.jku.at/publications/older/3304.pdf)), the idea of adding gradient norm penalty seems to be a relatively recent suggestion[(Implicit graidnet regularization)](https://arxiv.org/abs/2009.11162), [(Explicit gradient regularization)](https://arxiv.org/abs/2202.03599).
