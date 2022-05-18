# Design of Experiment
The conditions to be provided for each problem solution is:
1. Scheme used to apply EBC
	1. "Weighted" applies a weighting function to the interior point wrt the distance from boundary. Requires calculation at all points in the dataset
	2. "Dataset" applies EBC to a cloud of collocation points near the boundaries. Requires lesser calculations but attention to generating this collocation cloud is needed.
2. Choice of Basis Function
	1. Taylor's Series
	2. Lagrange Polynomial
	3. Trigonometric (Fourier)
	4. Generic Approximator - (Exponential)
3. Extent of Application
	1. Initial only - Starts using EBC but after an appreciable amount of convergence has begun, shifts to the original Boundary Conditions
	2. Always - Uses EBC as boundary till the end. Needs higher attention as error in EBC acts as a limit to error in solution.

Classes of problems under consideration
1. One Dimensional BVP
2. Higher Order Eigenvalue Problem
3. Singular Perturbation problem

Hence total number of experiments
$$
3\times (1 + 2\times4\times2) = 51
$$
