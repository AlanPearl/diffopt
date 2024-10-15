---
title: 'DiffOpt: Parallel optimization of Jax models'
tags:
  - Python
  - Jax
  - MPI
authors:
  - name: Alan N. Pearl
    orcid: 0000-0001-9820-9619
    affiliation: 1
  - name: Gillian D. Beltz-Mohrmann
    orcid: 0000-0002-4392-8920
    affiliation: 1
  - name: Andrew P. Hearin
    orcid: 0000-0003-2219-6852
    affiliation: 1
affiliations:
 - name: HEP Division, Argonne National Laboratory, 9700 South Cass Avenue, Lemont, IL 60439, USA
   index: 1
date: 9 October 2024
bibliography: paper.bib
---

# Summary

`diffopt` is a Python package which facilitates in the optimization of 
data-parallelized, differentiable models using the Jax [@jax2018github] 
framework. It is composed of three subpackages, `multigrad`, `kdescent`, and 
`multiswarm`. Leveraging MPI (Message Passing Interface), `multigrad` 
efficiently sums and propagates gradients of custom-defined summary statistics 
across processors and computing nodes. `kdescent` utilizes mini-batched kernel 
density estimates to perform stochastic gradient descent to fit a full model 
distribution to an N-dimensional training dataset. A massively parallelizable 
implementation of particle swarm optimization (PSO) is provided by 
`multiswarm`, enabling global optimization of even high-dimensional, 
non-convex loss surfaces. Our simple yet flexible design makes these methods 
applicable to a wide variety of problems requiring solutions scalable 
to large amounts of data through both gradient- and non-gradient-based 
optimization techniques. Visit our 
[documentation page](https://diffopt.readthedocs.io) to learn the usage.


# Statement of Need

In and beyond the field of cosmology, parameterized models can describe 
complex systems, provided that the parameters have been tuned adequately to 
fit the model to observational data. Fitting capabilities can be increased 
dramatically by gradient-based techniques, particularly in high-dimensional 
parameter spaces. Existing gradient descent tools in Jax do not inherently 
support data-parallelism with MPI, creating a speed and memory bottleneck 
for such computations.

`multigrad` addresses this need by providing an easy-to-use interface for 
implementing data-parallelized models. It handles the MPI reductions as well 
as the mathematical complexities involved in propagating chain rules required 
to compute the gradient of the loss, which is a function of parallelized 
summary statistics, which are in turn functions of the model parameters. 
At the same time, it is very flexible in that it allows users to define their 
own functions to compute their summary statistics and loss. As a result, this 
package can enable scalability through parallelization to the optimization 
routine of nearly any big-data model. `kdescent` and `multiswarm` each provide 
powerful fitting tools which are fully compatible with the parallelization 
framework laid out by `multigrad`.

# Method

## `multigrad`

`multigrad` allows the user to implement a loss term, which is a function of 
summary statistics, which are functions of parameters, $L(\vec{y}(\vec{x}))$ 
where the summary statistics are summed over multiple MPI-linked processes: 
$\vec{y} = \sum_i\vec{y}_{(i)}$ where $i$ is the index of each process. In 
this section, we will derive the gradient of the loss $\vec{\nabla} L$ with 
respect to the parameters and as a sum of terms that each process can compute 
independently.

We will begin from the definition of the multivariate chain rule,

$$ \frac{\partial L}{\partial x_j} = \sum\limits_{k} \frac{\partial L}
{\partial y_k} \frac{\partial y_k}{\partial x_j} $$

where $\partial y_k$ = $\sum_i\partial y_{k (i)}$. By pulling out the MPI 
summation over $i$,

$$ \frac{\partial L}{\partial x_j} = \sum\limits_{i} \sum\limits_{k} 
\frac{\partial L}{\partial y_k} \frac{\partial y_{k (i)}}{\partial x_j} $$

and by rewriting this as vector-matrix multiplication,

$$ \vec{\nabla_x} L = \sum\limits_{i} (\vec{\nabla_y} L)^T J_{(i)} $$

we can clearly identify that each process has to perform a vector-Jacobian 
product (VJP), where $J_{(i)}$ is the Jacobian matrix such that 
$J_{kj (i)} = \frac{\partial y_{k (i)}}{\partial x_j}$. Fortunately, this is a 
computation that Jax can perform very efficiently, without the need to 
explicitly calculate the full Jacobian matrix by making use of the
[`jax.vjp`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vjp.html) 
feature, saving us orders of magnitude of time and memory requirements.

## `kdescent`

Mini-batching techniques often compute the loss function with only a small 
subset of the training data taken into account. In `kdescent`, the density of 
the full training dataset is measured around a "mini-batched" sample of kernel 
centers, which are drawn from points in the training data. With each iteration 
of stochastic gradient descent, a new sample of (20 by default) kernels is 
selected at positions $\vec{\mu}_k$ for each kernel $k$.

Using the `compare_kde_counts` method, the "true" and "model" counts are each 
computed around each kernel using the same equation below, where $x_i$ is the 
$i^{\rm th}$ point in the training data or model data, respectively:

$$ N_k = \sum\limits_i \mathcal{N}(\vec{x}_i \; | \; \vec{\mu}_k, \Sigma) $$

where $\mathcal{N}$ is the multivariate-normal distribution with mean 
$\vec{\mu}_k$ and covariance matrix $\Sigma$ (where the covariance is 
calculated using Scott's rule for kernel density estimation of the training 
dataset; @Scott:1992). It is then up to the user to define their own loss 
function comparing the counts of $N_{k, \rm truth}$ to $N_{k, \rm model}$. 
Note that these are extrinsic quantities (as is necessary to be parallelizable 
through `multigrad`) which can be reduced to intrinsic quantities for 
PDF-level comparisons by simply dividing by the total number of training and 
model data, respectively.

The analogous `compare_fourier_counts` method can provide additional loss 
terms relating to differences in the empirical characteristic function (ECF; 
@Cramer:1954). It is evaluated at a random sample of (20 by default) 
fourier-space positions, $\vec{\tilde{x}}_k$, for both the "true" and "model" 
fourier counts:

$$ \tilde{N}_k = \sum\limits_i \exp(i \vec{\tilde{x}}_k \cdot \vec{x}_i). $$

## `multiswarm`

Particle swarm optimization (PSO; @Kennedy:1995) is a highly exploratory 
fitting algorithm in which a set of (100 by default) particles are initialized 
with randomized velocities and positions with Latin-Hypercube spacing over the 
loss function's parameter space. Each particle has an inertial weight 
($w_I = 1$ by default), a cognitive weight, ($w_C = 0.21$ by default), and a 
social weight, ($w_S = 0.07$ by default). The default parameters have been 
hand-tuned to optimize parameter exploration performed by 100 particles before 
converging over roughly 100 time steps in a 4D Ackley loss function 
[demonstrated in our documentation](
  https://diffopt.readthedocs.io/en/latest/multiswarm/intro.html).

Within each PSO iteration: (1) Each particle's position is updated according 
to its current velocity $x_{i+1} = x_i + v_i$. (2) Positions and velocities 
are then reflected accordingly across any axes in which they have left the 
boundaries, if applicable. (3) Finally, the particle's velocity is slightly 
pulled in the direction of its personal best $x_{\rm PB}$ and global best 
$x_{\rm GB}$ loss found, according to the following equation:

$$ v_{i+1} = w_I v_i + w_C (x_{\rm PB} - x_{i+1}) 
+ w_S (x_{\rm GB} - x_{i+1}) $$

The `multiswarm` implementation of PSO allows users to conveniently distribute 
the loss function computations performed by each particle across MPI ranks. 
Particles are evenly distributed across all ranks by default, but users 
needing further control can provide a custom MPI communicator object, and/or 
specify the `ranks_per_particle` argument to manually control intra-particle 
parallelization.

# Science Use Case

`diffopt` was developed to aid in parameter optimization for high-dimensional 
differentiable models applied to large datasets. It has enabled the scaling to 
cosmological volumes of a differentiable forward modeling pipeline which 
predicts galaxy properties based on a simulated dark matter density field 
(Diffmah: @Hearin:2021; Diffstar: @Alarcon:2023; DSPS: @Hearin:2023). Ongoing 
research is currently utilizing `diffopt` to optimize the parameters of this 
pipeline to reproduce observed galaxy properties (e.g. Beltz-Mohrmann et al. 
in prep.). More broadly, `diffopt` has useful applications for any scientific 
research that focuses on fitting high-dimensional models to large datasets and 
would benefit from computing parameter gradients in parallel.

# Acknowledgements
This work was supported in part by the OpenUniverse effort, which is funded by 
NASA under JPL Contract Task 70-711320, 'Maximizing Science Exploitation of 
Simulated Cosmological Survey Data Across Surveys', and by the DOE contract 
DE-AC02-06CH11357. We gratefully acknowledge the HPC resources operated by the 
Laboratory Computing Resource Center at Argonne National Laboratory.

# References
