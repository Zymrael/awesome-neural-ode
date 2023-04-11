
# Awesome Neural ODE

A collection of resources regarding the interplay between differential equations, dynamical systems, deep learning, control, numerical methods and scientific machine learning.

**NOTE:** Feel free to suggest additions via `Issues` or `Pull Requests`.

The repo further introduces a (rough) categorization by assigning topic labels to each work. These are not supposed to be comprehensive or precise, and should only provide a rough idea of the contents.

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)
![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)
![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)
![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)
![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

# Table of Contents

* **Differential Equations in Deep Learning**

	* [General Architectures](#general-architectures)
	
	* [Neural Operators](#neural-operators)
	
	* [Neural ODEs](#neural-odes)
	
		* [Training of Neural ODEs](#training-of-neural-odes)
		
		* [Speeding up continuous models](#speeding-up-continuous-models)
		
		* [Control with Neural ODEs](#control-with-neural-odes)
	
	* [Neural GDEs](#neural-gdes)
	
	* [Neural SDEs](#neural-sdes)
	
	* [Neural CDEs](#neural-cdes)
	
	* [Generative Models](#generative-models)
	
		* [Normalizing Flows](#normalizing-flows)
		
		* [Score-Matching SDEs](#score-matching-sdes) 	
	
	* [Applications](#applications)
	
* **Deep Learning Methods for Differential Equations (Scientific ML)**

	* [Solving Differential Equations](#solving-differential-equations)
	
	* [Model Discovery](#model-discovery)
	
* **Dynamical System View of Deep Learning**

	* [Recurrent Neural Networks](#recurrent-neural-networks)
	
	* [Theory and Perspectives](#theory-and-perspectives)
	
	* [Optimization](#optimization)
	
* [Software and Libraries](#software-and-libraries)

* [Websites and Blogs](#websites-and-blogs)

## Differential Equations in Deep Learning

### General Architectures

* Recurrent Neural Networks for Multivariate Time Series with Missing Values: [Scientific Reports18](https://arxiv.org/abs/1606.01865)

![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> Multivariate time series data in practical applications, such as health care, geoscience, and biology, are characterized by a variety of missing values. We propose a GRU-based model called GRU-D, in which a decay mechanism is designed for the input variables and the hidden states to capture the aforementioned properties. We introduce decay rates in the model to control the decay mechanism by considering the following important factors.

* Learning unknown ODE models with Gaussian processes: [arXiv18](https://arxiv.org/abs/1803.04303), [code](https://github.com/cagatayyildiz/npde/)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> However, for many complex systems it is practically impossible to determine the equations or
interactions governing the underlying dynamics. In these settings, parametric ODE model cannot be formulated. Here, we overcome this issue by introducing a novel paradigm of nonparametric ODE modeling that can learn the underlying dynamics of arbitrary continuous-time systems without prior knowledge. We propose to learn non-linear, unknown differential functions from state observations using Gaussian process vector fields within the exact ODE formalism.

* Deep Equilibrium Models: [NeurIPS19](https://arxiv.org/abs/1909.01377)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> We present a new approach to modeling sequential data: the deep equilibrium model (DEQ). Motivated by an observation that the hidden layers of many existing deep sequence models converge towards some fixed point, we propose the DEQ approach that directly finds these equilibrium points via root-finding.

* Fast and Deep Graph Neural Networks: [AAAI20](https://arxiv.org/pdf/1911.08941.pdf)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> We address the efficiency issue for the construction of a deep graph neural network (GNN). The approach exploits the idea of representing each input graph as a fixed point of a dynamical system (implemented through a recurrent neural network), and leverages a deep architectural organization of the recurrent units. Efficiency is gained by many aspects, including the use of small and very sparse networks, where the weights of the recurrent units are left untrained under the stability condition introduced in this work.

* Hamiltonian Neural Networks: [NeurIPS19](https://arxiv.org/abs/1906.01563)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> In this paper, we draw inspiration from Hamiltonian mechanics to train models that learn and respect exact conservation laws in an unsupervised manner.

* Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning: [ICLR19](https://arxiv.org/abs/1907.04490)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> We propose Deep Lagrangian Networks (DeLaN) as a deep network structure upon which Lagrangian Mechanics have been imposed. DeLaN can learn the equations of motion of a mechanical system (i.e., system dynamics) with a deep network efficiently while ensuring physical plausibility. The resulting DeLaN network performs very well at robot tracking control.

* Lagrangian Neural Networks: [ICLR20 DeepDiffEq](https://arxiv.org/abs/2003.04630)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> We propose Lagrangian Neural Networks (LNNs), which can parameterize arbitrary Lagrangians using neural networks. In contrast to models that learn Hamiltonians, LNNs do not require canonical coordinates, and thus perform well in situations where canonical momenta are unknown or difficult to compute.

* Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints: [NeurIPS20](https://arxiv.org/abs/2010.13581), [code](https://github.com/mfinzi/constrained-hamiltonian-neural-networks)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

> Reasoning about the physical world requires models that are endowed with the right inductive biases to learn the underlying dynamics. Recent works improve generalization for predicting trajectories by learning the Hamiltonian or Lagrangian of a system rather than the differential equations directly. While these methods encode the constraints of the systems using generalized coordinates, we show that embedding the system into Cartesian coordinates and enforcing the constraints explicitly with Lagrange multipliers dramatically simplifies the learning problem.

### Neural Operators 

* Neural Operator: Learning Maps Between Function Spaces: [arXv21](https://arxiv.org/abs/2108.08481)

> We propose a generalization of neural networks to learn operators that maps between infinite dimensional function spaces. We formulate the approximation of operators by composition of a class of linear integral operators and nonlinear activation functions, so that the composed operator can approximate complex nonlinear operators. We prove a universal approximation theorem for our construction. Furthermore, we introduce four classes of operator parameterizations: graph-based operators, low-rank operators, multipole graph-based operators, and Fourier operators and describe efficient algorithms for computing with each one.

* Fourier Neural Operator for Parametric Partial Differential Equations: [ICLR 2021](https://arxiv.org/abs/2010.08895)

> We formulate a new neural operator by parameterizing the integral kernel directly in Fourier space, allowing for an expressive and efficient architecture.

* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators

> FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that provides accurate short to medium-range global predictions at 0.25∘ resolution. FourCastNet accurately forecasts high-resolution, fast-timescale variables such as the surface wind speed, precipitation, and atmospheric water vapor.

* Transform Once: Efficient Operator Learning in Frequency Domain

> This work introduces a blueprint for frequency domain learning through a single transform: transform once (T1). To enable efficient, direct learning in the frequency domain we develop a variance preserving weight initialization scheme and address the open problem of choosing a transform. Our results noticeably streamline the design process of frequency-domain models, pruning redundant transforms, and leading to speedups of 3x to 10x that increase with data resolution and model size. We perform extensive experiments on learning to solve partial differential equations, including incompressible Navier-Stokes, turbulent flows around airfoils, and high-resolution video of smoke dynamics. T1 models improve on the test performance of SOTA FDMs while requiring significantly less computation, with over 20% reduction in predictive error across tasks.

### Neural ODEs

* Neural Ordinary Differential Equations (best paper award): [NeurIPS18](https://arxiv.org/pdf/1806.07366.pdf) 

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions

* Dissecting Neural ODEs (oral): [NeurIPS20](https://arxiv.org/abs/2002.08071) 

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool) ![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> Continuous deep learning architectures have recently re-emerged as *Neural Ordinary Differential Equations* (Neural ODEs). This infinite--depth approach theoretically bridges the gap between deep learning and dynamical systems, offering a novel perspective. However, deciphering the inner working of these models is still an open challenge, as most applications apply them as generic *black--box* modules. In this work we "open the box", further developing the continuous-depth formulation with the aim of clarifying the influence of several design choices on the underlying dynamics. 

* Differentiable Multiple Shooting Layers: [NeurIPS21](https://arxiv.org/abs/2106.03885)

> We detail a novel class of implicit neural models. Leveraging time-parallel methods for differential equations, Multiple Shooting Layers (MSLs) seek solutions of initial value problems via parallelizable root-finding algorithms. MSLs broadly serve as drop-in replacements for neural ordinary differential equations (Neural ODEs) with improved efficiency in number of function evaluations (NFEs) and wall-clock inference time.

* Augmented Neural ODEs: [NeurIPS19](https://arxiv.org/abs/1904.01681) 

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> We show that Neural Ordinary Differential Equations (ODEs) learn representations that preserve the topology of the input space and prove that this implies the existence of functions Neural ODEs cannot represent. To address these limitations, we introduce Augmented Neural ODEs which, in addition to being more expressive models, are empirically more stable, generalize better and have a lower computational cost than Neural ODEs.

* Latent ODEs for Irregularly-Sampled Time Series: [NeurIPS19](https://arxiv.org/abs/1907.03907)

![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

* ODE2VAE: Deep generative second order ODEs with Bayesian neural networks: [NeurIPS19](https://arxiv.org/pdf/1905.10994.pdf)

![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

* Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control: [arXiv19](https://arxiv.org/abs/1909.12077)

* Stable Neural Flows: [arXiv20](https://arxiv.org/abs/2003.08063) 

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool)

* On Second Order Behaviour in Augmented Neural ODEs [NeurIPS20](https://arxiv.org/abs/2006.07220)

![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

* Neural Hybrid Automata: Learning Dynamics with Multiple Modes and Stochastic Transitions: [NeurIPS21](https://arxiv.org/abs/2106.04165)

> Effective control and prediction of dynamical systems often require appropriate handling of continuous-time and discrete, event-triggered processes. Stochastic hybrid systems (SHSs), common across engineering domains, provide a formalism for dynamical systems subject to discrete, possibly stochastic, state jumps and multi-modal continuous-time flows. Despite the versatility and importance of SHSs across applications, a general procedure for the explicit learning of both discrete events and multi-mode continuous dynamics remains an open problem. This work introduces Neural Hybrid Automata (NHAs), a recipe for learning SHS dynamics without a priori knowledge on the number of modes and inter-modal transition dynamics. NHAs provide a systematic inference method based on normalizing flows, neural differential equations and self-supervision.

#### Training of Neural ODEs

* Accelerating Neural ODEs with Spectral Elements: [arXiv19](https://arxiv.org/abs/1906.07038) 

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

* Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE: [ICML20](https://arxiv.org/abs/2006.02493) 

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor) ![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

* MALI: A memory efficient and reverse accurate integrator for Neural ODEs: [ICLR21](https://openreview.net/pdf?id=blfSjHeFM_e) 

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor) ![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> Existing implementations of the adjoint method suffer from inaccuracy in reverse-time trajectory, while the naive method and the adaptive checkpoint adjoint method (ACA) have a memory cost that grows with integration time. In this project, based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost w.r.t number of solver steps in integration similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation).

#### Speeding up continuous models

* How to Train you Neural ODE: [ICML20](https://arxiv.org/abs/2002.02798)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

* Learning Differential Equations that are Easy to Solve: [NeurIPS20](https://arxiv.org/abs/2007.04504) 

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

* Hypersolvers: Toward Fast Continuous-Depth Models: [NeurIPS20](https://arxiv.org/abs/2007.09601) 

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

* Hey, that's not an ODE": Faster ODE Adjoints with 12 Lines of Code: [arXiV20](https://arxiv.org/pdf/2009.09457.pdf) 

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

> Neural differential equations may be trained by backpropagating gradients via the adjoint method. Here, we demonstrate that the particular structure of the adjoint equations makes the usual choices of norm (such as L2) unnecessarily stringent. By replacing it with a more appropriate (semi)norm, fewer steps are unnecessarily rejected and the backpropagation is made faster.

* Interpolation Technique to Speed Up Gradients Propagation in Neural ODEs: [NeurIPS20](https://papers.nips.cc/paper/2020/file/c24c65259d90ed4a19ab37b6fd6fe716-Paper.pdf)

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor) ![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> We propose a simple interpolation-based method for the efficient approximation of gradients in neural ODE models. We compare it with the reverse dynamic method (known in the literature as “adjoint method”) to train neural ODEs on classification, density estimation, and inference approximation tasks.

* Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics: [ICML21](https://arxiv.org/abs/2105.03918)

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

> Can we force the NDE to learn the version with the least steps while not increasing the training cost? Current strategies to overcome slow prediction require high order automatic differentiation, leading to significantly higher training time. We describe a novel regularization method that uses the internal cost heuristics of adaptive differential equation solvers combined with discrete adjoint sensitivities

#### Control with Neural ODEs

* Model-based Reinforcement Learning for Semi-Markov Decision Processes with Neural ODEs: [NeurIPS20](https://arxiv.org/pdf/2006.16210.pdf)

> In this paper, we take a model-based approach to continuous-time RL, modeling the dynamics via neural ordinary differential equations (ODEs). Not only is this more sample efficient than model-free approaches, but it allows us to efficiently adapt policies learned using one schedule of interactions with the environment for another.

* Optimal Energy Shaping via Neural Approximators: [arXiv20](https://arxiv.org/abs/2101.05537)

> We introduce optimal energy shaping as an enhancement of classical passivity-based control methods. A promising feature of passivity theory, alongside stability, has traditionally been claimed to be intuitive performance tuning along the execution of a given task. However, a systematic approach to adjust performance within a passive control framework has yet to be developed, as each method relies on few and problem-specific practical insights. Here, we cast the classic energy-shaping control design process in an optimal control framework; once a task-dependent performance metric is defined, an optimal solution is systematically obtained through an iterative procedure relying on neural networks and gradient-based optimization.

### Neural GDEs

* Graph Neural Ordinary Differential Equations (spotlight): [AAAI DLGMA20](https://arxiv.org/abs/1911.07532)

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> We introduce the framework of continuous–depth graph neural networks (GNNs). Neural graph ordinary differential equations (Neural GDEs) are formalized as the counterpart to GNNs where the input–output relationship is determined by a continuum of GNN layers, blending discrete topological structures and differential equations. We further introduce general Hybrid Neural GDE models as a hybrid dynamical systems. 

* Continuous–Depth Neural Models for Dynamic Graph Prediction: [arXiv21](https://arxiv.org/pdf/2106.11581.pdf), extended version of "Graph Neural Ordinary Differential Equations"

![DS](https://img.shields.io/badge/systems-red.svg?logo=Graphcool) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> Additional Neural GDE variants are developed to tackle the spatio–temporal setting of dynamic graphs. The evaluation protocol for Neural GDEs spans several application domains, including traffic forecasting and prediction in biological networks.

* GRAND: Graph Neural Diffusion:  [arXiv21](https://arxiv.org/abs/2106.10934)

> We present Graph Neural Diffusion (GRAND) that approaches deep learning on graphs as a continuous diffusion process and treats Graph Neural Networks (GNNs) as discretisations of an underlying PDE

### Neural SDEs

* Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise: [arXiv19](https://arxiv.org/abs/1906.02355)

* Neural Jump Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1905.10403)

![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

* Towards Robust and Stable Deep Learning Algorithms for Forward Backward Stochastic Differential Equations: [arXiv19](https://arxiv.org/abs/1910.11623)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Scalable Gradients and Variational Inference for Stochastic Differential Equations: [AISTATS20](https://arxiv.org/abs/2001.01328)

* Score-Based Generative Modeling through Stochastic Differential Equations (oral): [ICLR20](https://openreview.net/pdf?id=PxTIG12RRHS)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise.

* Efficient and Accurate Gradients for Neural SDEs: [NeurIPS21](https://arxiv.org/abs/2105.13493)

> we introduce the reversible Heun method. This is a new SDE solver that is algebraically reversible: eliminating numerical gradient errors, and the first such solver of which we are aware. Moreover it requires half as many function evaluations as comparable solvers, giving up to a 1.98× speedup. Second, we introduce the Brownian Interval: a new, fast, memory efficient, and exact way of sampling \textit{and reconstructing} Brownian motion.


### Neural CDEs

* Neural Controlled Differential Equations for Irregular Time Series (spotlight): [NeurIPS20](https://arxiv.org/abs/2005.08926)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> We demonstrate how controlled differential equations may extend the Neural ODE model, which we refer to as the neural controlled differential equation (Neural CDE) model. Just as Neural ODEs are the continuous analogue of a ResNet, the Neural CDE is the continuous analogue of an RNN.

* Neural CDEs for Long Time Series via the Log-ODE Method: [arXiv20](https://arxiv.org/abs/2009.08295)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

* Neural Controlled Differential Equations for Online Prediction Tasks: [arXiv21](https://arxiv.org/abs/2106.11028)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz) ![TS](https://img.shields.io/badge/sequences-purple.svg?logo=Altium%20Designer)

> We identify several theoretical conditions that interpolation schemes for Neural CDEs should satisfy, such as boundedness and uniqueness. Second, we use these to motivate the introduction of new schemes that address these conditions, offering in particular measurability (for online prediction), and smoothness (for speed).

### Generative Models

#### Normalizing Flows

* Monge-Ampère Flow for Generative Modeling: [arXiv18](https://arxiv.org/pdf/1809.10188.pdf)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

* FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models: [ICLR19](https://arxiv.org/abs/1810.01367)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

* Equivariant Flows: sampling configurations for multi-body systems with symmetric energies: [arXiv18](https://arxiv.org/pdf/1910.00753.pdf)

* Flows for simultaneous manifold learning and density estimation: [NeurIPS20](https://arxiv.org/abs/2003.13913)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

> We introduce manifold-learning flows (M-flows), a new class of generative models that simultaneously learn the data manifold as well as a tractable probability density on that manifold. We argue why such models should not be trained by maximum likelihood alone and present a new training algorithm that separates manifold and density updates.

* TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics [arXiv20](https://arxiv.org/pdf/2002.04461.pdf)

* Convex Potential Flows: Universal Probability Distributions with Optimal Transport and Convex Optimization: [arXiv20](https://arxiv.org/pdf/2012.05942.pdf)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> CP-Flows are the gradient map of a strongly convex neural potential function. The convexity implies invertibility and allows us to resort to convex optimization to solve the convex conjugate for efficient inversion.

#### Diffusion Models

* Score-Based Generative Modeling through Stochastic Differential Equations (best paper award): [ICLR21](https://openreview.net/pdf?id=PxTIG12RRHS)

![IC](https://img.shields.io/badge/images-blue.svg?logo=Google%20Classroom)

> Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. 

* Denoising Diffusion Implicit Models

> Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process.

### Applications 

* Learning Dynamics of Attention: Human Prior for Interpretable Machine Reasoning: [NeurIPS19](https://arxiv.org/abs/1905.11666)

## Deep Learning Methods for Differential Equations

### Solving Differential Equations

* PDE-Net: Learning PDEs From Data: [ICML18](https://arxiv.org/abs/1710.09668)

### Model Discovery

* Universal Differential Equations for Scientific Machine Learning: [arXiv20](https://arxiv.org/abs/2001.04385)

![NM](https://img.shields.io/badge/numerics-green.svg?logo=CodeFactor)

## Dynamical System View of Deep Learning

### Recurrent Neural Networks

* A Comprehensive Review of Stability Analysis of Continuous-Time Recurrent Neural Networks: [IEEE Transactions on Neural Networks 2006](https://ieeexplore.ieee.org/abstract/document/6814892)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* AntysimmetricRNN: A Dynamical System View on Recurrent Neural Networks: [ICLR19](https://openreview.net/pdf?id=ryxepo0cFX)

* Recurrent Neural Networks in the Eye of Differential Equations: [arXiv19](https://arxiv.org/pdf/1904.12933.pdf)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Visualizing memorization in RNNs: [distill19](https://distill.pub/2019/memorization-in-rnns/)

* One step back, two steps forward: interference and learning in recurrent neural networks: [arXiv18](https://arxiv.org/abs/1805.09603)

* Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics: [arXiv19](https://arxiv.org/pdf/1906.10720.pdf)

* System Identification with Time-Aware Neural Sequence Models: [AAAI20](https://arxiv.org/abs/1911.09431)

* Universality and Individuality in recurrent networks: [NeurIPS19](https://arxiv.org/abs/1907.08549)

### Theory and Perspectives

* A Proposal on Machine Learning via Dynamical Systems: [Communications in Mathematics and Statistics 2017](https://link.springer.com/content/pdf/10.1007/s40304-017-0103-z.pdf)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Deep Learning Theory Review: An Optimal Control and Dynamical Systems Perspective: [arXiv19](https://arxiv.org/abs/1908.10920)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Stable Architectures for Deep Neural Networks: [IP17](https://arxiv.org/pdf/1705.03341.pdf)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Beyond Finite Layer Neural Network: Bridging Deep Architects and Numerical Differential Equations: [ICML18](https://arxiv.org/abs/1710.10121)

* Review: Ordinary Differential Equations For Deep Learning: [arXiv19](https://arxiv.org/abs/1911.00502)

### Optimization

* Gradient and Hamiltonian Dynamics Applied to Learning in Neural Networks: [NIPS96](https://papers.nips.cc/paper/1033-gradient-and-hamiltonian-dynamics-applied-to-learning-in-neural-networks.pdf)

* Maximum Principle Based Algorithms for Deep Learning: [JMLR17](https://arxiv.org/abs/1710.09513)

* Hamiltonian Descent Methods: [arXiv18](https://arxiv.org/pdf/1809.05042.pdf)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* Port-Hamiltonian Approach to Neural Network Training: [CDC19](https://arxiv.org/abs/1909.02702), [code](https://github.com/Zymrael/PortHamiltonianNN)

![T](https://img.shields.io/badge/theory-black.svg?logo=MusicBrainz)

* An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks: [arXiv19](https://arxiv.org/abs/1803.01299)

* Optimizing Millions of Hyperparameters by Implicit Differentiation: [arXiv19](https://arxiv.org/abs/1911.02590)

* Shadowing Properties of Optimization Algorithms: [NeurIPS19](https://papers.nips.cc/paper/9431-shadowing-properties-of-optimization-algorithms)

## Software and Libraries

### Python

* **torchdyn**: PyTorch library for all things neural differential equations. [repo](https://github.com/diffeqml/torchdyn), [docs](https://torchdyn.readthedocs.io/)
* **torchdiffeq**: Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation: [repo](https://github.com/rtqichen/torchdiffeq)
* **torchsde**: Stochastic differential equation (SDE) solvers with GPU support and efficient sensitivity analysis: [repo](https://github.com/google-research/torchsde)
* **torchcde**: GPU-capable solvers for controlled differential equations (CDEs): [repo](https://github.com/patrick-kidger/torchcde)
* **torchSODE**: PyTorch Block-Diagonal ODE solver: [repo](https://github.com/Zymrael/torchSODE)
* **neurodiffeq**: A light-weight & flexible library for solving differential equations using neural networks based on PyTorch: [repo](https://github.com/NeuroDiffGym/neurodiffeq)

### Julia

* **DiffEqFlux**: [repo](https://github.com/JuliaDiffEq/DiffEqFlux.jl)

> Neural differential equation solvers with O(1) backprop, GPUs, and stiff+non-stiff DE solvers. Supports stiff and non-stiff neural ordinary differential equations (neural ODEs), neural stochastic differential equations (neural SDEs), neural delay differential equations (neural DDEs), neural partial differential equations (neural PDEs), and neural jump stochastic differential equations (neural jump diffusions). All of these can be solved with high order methods with adaptive time-stepping and automatic stiffness detection to switch between methods. 

* **NeuralNetDiffEq**: Implementations of ODE, SDE, and PDE solvers via deep neural networks: [repo](https://github.com/JuliaDiffEq/NeuralNetDiffEq.jl)

## Websites and Blogs

* Scientific ML Blog (Chris Rackauckas and SciML): [link](http://www.stochasticlifestyle.com/)
