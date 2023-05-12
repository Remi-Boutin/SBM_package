# SBM package

## Installation

To install this package, run 

``python -m pip install sbm-vbem``


## Quick use of the SBM function

In this package, I implemented 2 different estimations : *a variational-bayes EM (it's on the way)* and a variational-EM.
The variational-bayes EM can be optimized in two ways : setting the lagrangian to zero, 
referred to as the **VBEM** algorithm, or using a natural-conjugate gradient method,
referred to as the **NCG** algorithm.

The main function ``sbm`` is only implemented for directed graphs (for now) and can be used this way:

- with no init given : ``sbm(adj, algo='vbem', init='kmeans') ``
- with an already computed init : ``sbm(adj, algo='vbem', tau_init=tau) ``

This is a joint work with Pr. Pierre Latouche and Pr. Charles Bouveyron.