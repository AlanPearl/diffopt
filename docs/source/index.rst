.. diffopt documentation master file, created by
   sphinx-quickstart on Fri Oct 4 10:21:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to diffopt's documentation!
===================================

DiffOpt is a collection of tools that are useful for parallelizable
optimization of large-parameter, memory-intensive, and/or differentiable
models implemented in Jax. It is composed of the following three subpackages:
(1) ``multigrad`` enables enables you to define a data-parallelized (over MPI)
loss function and compute its gradient, (2) ``kdescent`` performs stochastic
gradient descent over mini-batched KDE statistics, and (3) ``multiswarm`` is an
MPI-parallelized implementation of Particle Swarm Optimization (PSO).
The code is open-source and available on
`GitHub <https://github.com/AlanPearl/diffopt>`__.

Overview
--------

* :doc:`installation`
* :doc:`reference`

:mod:`multigrad`
================

* :doc:`multigrad/intro`

:mod:`kdescent`
===============

* :doc:`kdescent/intro`
* :doc:`kdescent/hmf_upweight`
* :doc:`kdescent/integration`

:mod:`multiswarm`
=================

* :doc:`multiswarm/intro`

Indices and tables
------------------

* :ref:`genindex`

.. * :ref:`search`
.. * :ref:`modindex`

.. Hidden TOCs

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:

   installation.rst
   reference.rst


.. toctree::
   :maxdepth: 2
   :caption: MULTIGRAD
   :hidden:

   multigrad/intro.ipynb


.. toctree::
   :maxdepth: 2
   :caption: KDESCENT
   :hidden:

   kdescent/intro.ipynb
   kdescent/hmf_upweight.ipynb
   kdescent/integration.ipynb


.. toctree::
   :maxdepth: 2
   :caption: MULTISWARM
   :hidden:

   multiswarm/intro.ipynb