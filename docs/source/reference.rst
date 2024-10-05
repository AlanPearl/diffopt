.. :tocdepth: 3

*************
API Reference
*************

.. contents:: Table of Contents

:mod:`multigrad`
================

.. autoclass:: diffopt.multigrad.OnePointModel
    :members:

.. autoclass:: diffopt.multigrad.OnePointGroup
    :members:

.. autofunction:: diffopt.multigrad.split_subcomms

.. autofunction:: diffopt.multigrad.split_subcomms_by_node

.. autofunction:: diffopt.multigrad.reduce_sum


:mod:`kdescent`
===============

.. autoclass:: diffopt.kdescent.KCalc
    :members:
    :special-members: __init__

.. autofunction:: diffopt.kdescent.adam

.. autofunction:: diffopt.kdescent.bfgs


:mod:`multiswarm`
=================

.. autoclass:: diffopt.multiswarm.ParticleSwarm
    :members:
    :special-members: __init__

.. autofunction:: diffopt.multiswarm.get_best_loss_and_params

.. autofunction:: diffopt.multiswarm.get_subcomm
