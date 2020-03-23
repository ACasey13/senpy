# senpy
A python package of algorithms for sensitivity testing.

Currently implements the Neyer method which consists of:
  - Using maximum likelihood estimators (MLEs) to estimate the parameters of an assume latent distribution.
  - Provides a sequential design routine to suggest to the user new stimulus levels for efficent testing.
  
Future versions will add the following functionality: 
  - Implement the Dror-Steinberg method. (Bayesian approach)
  - Include a Gaussian process classifier with monitonicity constraint.
  - Add ability to evaluate multivariate systems. 
