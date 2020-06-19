# awesome-ode-neural-networks
A collection of resources regarding the interplay between ODEs, dynamical systems, spatio-temporal modeling and deep learning.


## Architectures

* Recurrent Neural Networks for Multivariate Time Series with Missing Values: [Scientific Reports18](https://arxiv.org/abs/1606.01865)

* Learning unknown ODE models with Gaussian processes: [arXiv18](https://arxiv.org/abs/1803.04303)

* Stable Architectures for Deep Neural Networks: [IP17](https://arxiv.org/pdf/1705.03341.pdf)

* Beyond Finite Layer Neural Network: Bridging Deep Architects and Numerical Differential Equations: [ICML18](https://arxiv.org/abs/1710.10121)

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

* Fast and Deep Graph Neural Networks: [AAAI20](https://arxiv.org/pdf/1911.08941.pdf)

* Hamiltonian Neural Networks: [NeurIPS19](https://arxiv.org/abs/1906.01563)

* Lagrangian Neural Networks: [ICLR20 DeepDiffEq](https://arxiv.org/abs/2003.04630)

### Parametric ODEs

* Neural Ordinary Differential Equations: [NeurIPS18](https://arxiv.org/pdf/1806.07366.pdf)

* Graph Neural Ordinary Differential Equations: [arXiv19](https://arxiv.org/abs/1911.07532)

* Augmented Neural ODEs: [NeurIPS19](https://arxiv.org/abs/1904.01681)

* Latent ODEs for Irregularly-Sampled Time Series: [arXiv19](https://arxiv.org/abs/1907.03907)

* ODE2VAE: Deep generative second order ODEs with Bayesian neural networks: [NeurIPS19](https://arxiv.org/pdf/1905.10994.pdf)

* Accelerating Neural ODEs with Spectral Elements: [arXiv19](https://arxiv.org/abs/1906.07038)

* Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control: [arXiv19](https://arxiv.org/abs/1909.12077)

* How to Train you Neural ODE: [arXiv20](https://arxiv.org/abs/2002.02798)

* Dissecting Neural ODEs: [arXiv20](https://arxiv.org/abs/2002.08071)

* Stable Neural Flows: [arXiv20](https://arxiv.org/abs/2003.08063)

* On Second Order Behaviour in Augmented Neural ODEs [arXiv20](https://arxiv.org/abs/2006.07220)

### SDEs

* Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise: [arXiv19](https://arxiv.org/abs/1906.02355)

* Neural Jump Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1905.10403)

* Towards Robust and Stable Deep Learning Algorithms for Forward Backward Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1910.11623)

* Scalable Gradients and Variational Inference for Stochastic Differential Equations: [AISTATS20](https://arxiv.org/abs/2001.01328)

### Continuous Normalizing Flows

* FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models: [ICLR19](https://arxiv.org/abs/1810.01367)

* Equivariant Flows: sampling configurations for multi-body systems with symmetric energies: [arXiv18](https://arxiv.org/pdf/1910.00753.pdf)

### PDEs

* PDE-Net: Learning PDEs From Data: [ICML18](https://arxiv.org/abs/1710.09668)

## Scientific Machine Learning

* Universal Differential Equations for Scientific Machine Learning: [arXiv20](https://arxiv.org/abs/2001.04385)

## RNNs as Dynamical Systems

* A Comprehensive Review of Stability Analysis of Continuous-Time Recurrent Neural Networks: [IEEE Transactions on Neural Networks 2006](https://ieeexplore.ieee.org/abstract/document/6814892)

* AntysimmetricRNN: A Dynamical System View on Recurrent Neural Networks: [ICLR19](https://openreview.net/pdf?id=ryxepo0cFX)

* Recurrent Neural Networks in the Eye of Differential Equations: [arXiv19](https://arxiv.org/pdf/1904.12933.pdf)

* Visualizing memorization in RNNs: [distill19](https://distill.pub/2019/memorization-in-rnns/)

* One step back, two steps forward: interference and learning in recurrent neural networks: [arXiv18](https://arxiv.org/abs/1805.09603)

* Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics: [arXiv19](https://arxiv.org/pdf/1906.10720.pdf)

* System Identification with Time-Aware Neural Sequence Models: [AAAI20](https://arxiv.org/abs/1911.09431)

* Universality and Individuality in recurrent networks: [NeurIPS19](https://arxiv.org/abs/1907.08549)

## Optimization

* Gradient and Hamiltonian Dynamics Applied to Learning in Neural Networks: [NIPS96](https://papers.nips.cc/paper/1033-gradient-and-hamiltonian-dynamics-applied-to-learning-in-neural-networks.pdf)

* Maximum Principle Based Algorithms for Deep Learning: [JMLR17](https://arxiv.org/abs/1710.09513)

* Hamiltonian Descent Methods: [arXiv18](https://arxiv.org/pdf/1809.05042.pdf)

* Port-Hamiltonian Approach to Neural Network Training: [CDC19](https://arxiv.org/abs/1909.02702), [code](https://github.com/Zymrael/PortHamiltonianNN)

* An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks: [arXiv19](https://arxiv.org/abs/1803.01299)

* Optimizing Millions of Hyperparameters by Implicit Differentiation: [arXiv19](https://arxiv.org/abs/1911.02590)

* Shadowing Properties of Optimization Algorithms: [NeurIPS19](https://papers.nips.cc/paper/9431-shadowing-properties-of-optimization-algorithms)

## Theory

* Deep Learning Theory Review: An Optimal Control and Dynamical Systems Perspective: [arXiv19](https://arxiv.org/abs/1908.10920)

* Review: Ordinary Differential Equations For Deep Learning: [arXiv19](https://arxiv.org/abs/1911.00502)

## Dynamical systems and neural differential equations for other areas

* Learning Dynamics of Attention: Human Prior for Interpretable Machine Reasoning: [NeurIPS19](https://arxiv.org/abs/1905.11666)

## Tools

* torchdyn: PyTorch library for all things neural differential equations. [repo](https://github.com/diffeqml/torchdyn), [docs](https://torchdyn.readthedocs.io/)

* DiffEqFlux: Neural differential equation solvers with O(1) backprop, GPUs, and stiff+non-stiff DE solvers. 
  Supports stiff and non-stiff neural ordinary differential equations (neural ODEs), neural stochastic differential 
  equations (neural SDEs), neural delay differential equations (neural DDEs), neural partial differential 
  equations (neural PDEs), and neural jump stochastic differential equations (neural jump diffusions).
  All of these can be solved with high order methods with adaptive time-stepping and automatic stiffness
  detection to switch between methods. [repo](https://github.com/JuliaDiffEq/DiffEqFlux.jl)
  
* NeuralNetDiffEq: Implementations of ODE, SDE, and PDE solvers via deep neural networks: [repo](https://github.com/JuliaDiffEq/NeuralNetDiffEq.jl)

* torchdiffeq: Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation: [repo](https://github.com/rtqichen/torchdiffeq)

* torchSODE: PyTorch Block-Diagonal ODE solver: [repo](https://github.com/Zymrael/torchSODE)

## Blogs 

* Scientific ML Blog (Dr Christopher Rackauckas): [link](http://www.stochasticlifestyle.com/)
