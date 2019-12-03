# awesome-ode-neural-networks
A collection of resources regarding the interplay between ODEs, dynamical systems, spatio-temporal modeling and deep learning.


## Architectures

* Recurrent Neural Networks for Multivariate Time Series with Missing Values: [Scientific Reports18](https://arxiv.org/abs/1606.01865)

* Learning unknown ODE models with Gaussian processes: [arXiv18](https://arxiv.org/abs/1803.04303)

* Stable Architectures for Deep Neural Networks: [IP17](https://arxiv.org/pdf/1705.03341.pdf)

* Beyond Finite Layer Neural Network:Bridging Deep Architects and Numerical Differential Equations: [ICML18](https://arxiv.org/abs/1710.10121)

### Parametric ODEs

* Neural Ordinary Differential Equations: [NeurIPS18](https://arxiv.org/pdf/1806.07366.pdf)

* Graph Neural Ordinary Differential Equations: [arXiv19](https://arxiv.org/abs/1911.07532)

* Augmented Neural ODEs: [arXiv19](https://arxiv.org/abs/1904.01681)

* Latent ODEs for Irregularly-Sampled Time Series: [arXiv19](https://arxiv.org/abs/1907.03907)

* ODE2VAE: Deep generative second order ODEs with Bayesian neural networks: [NeurIPS19](https://arxiv.org/pdf/1905.10994.pdf)

* Accelerating Neural ODEs with Spectral Elements: [arXiv19](https://arxiv.org/abs/1906.07038)

* Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control: [arXiv19](https://arxiv.org/abs/1909.12077)

### SDEs

* Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise: [arXiv19](https://arxiv.org/abs/1906.02355)

* Neural Jump Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1905.10403)

* Towards Robust and Stable Deep Learning Algorithms for Forward Backward Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1910.11623)

## RNNs as Dynamical Systems

* AntysimmetricRNN: A Dynamical System View on Recurrent Neural Networks: [ICLR19](https://openreview.net/pdf?id=ryxepo0cFX)

* Recurrent Neural Networks in the Eye of Differential Equations: [arXiv19](https://arxiv.org/pdf/1904.12933.pdf)

* Visualizing memorization in RNNs: [distill19](https://distill.pub/2019/memorization-in-rnns/)

* One step back, two steps forward: interference and learning in recurrent neural networks: [arXiv18](https://arxiv.org/abs/1805.09603)

* Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics: [arXiv19](https://arxiv.org/pdf/1906.10720.pdf)

* System Identification with Time-Aware Neural Sequence Models: [AAAI20](https://arxiv.org/abs/1911.09431)

## ODE-Based Optimization

* Gradient and Hamiltonian Dynamics Applied to Learning in Neural Networks: [NIPS96](https://papers.nips.cc/paper/1033-gradient-and-hamiltonian-dynamics-applied-to-learning-in-neural-networks.pdf)

* Hamiltonian Descent Methods: [arXiv18](https://arxiv.org/pdf/1809.05042.pdf)

* Port-Hamiltonian Approach to Neural Network Training: [CDC19](https://arxiv.org/abs/1909.02702), [code](https://github.com/Zymrael/PortHamiltonianNN)

## Theory

* Deep Learning Theory Review: An Optimal Control and Dynamical Systems Perspective: [arXiv19](https://arxiv.org/abs/1908.10920)

## Tools

* DiffEqFlux: Neural differential equation solvers with O(1) backprop, GPUs, and stiff+non-stiff DE solvers. 
  Supports stiff and non-stiff neural ordinary differential equations (neural ODEs), neural stochastic differential 
  equations (neural SDEs), neural delay differential equations (neural DDEs), neural partial differential 
  equations (neural PDEs), and neural jump stochastic differential equations (neural jump diffusions).
  All of these can be solved with high order methods with adaptive time-stepping and automatic stiffness
  detection to switch between methods. [repo](https://github.com/JuliaDiffEq/DiffEqFlux.jl)
  
* NeuralNetDiffEq: Implementations of ODE, SDE, and PDE solvers via deep neural networks: [repo](https://github.com/JuliaDiffEq/NeuralNetDiffEq.jl)

* torchdiffeq: Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation: [repo](https://github.com/rtqichen/torchdiffeq)

* torchSODE: PyTorch Block-Diagonal ODE solver: [repo](https://github.com/Zymrael/torchSODE)


