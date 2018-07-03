Sequential Monte Carlo
======================

We need to implement sequential monte carlo (SMC) in order to sufficiently speed up the ABC fitting process for regular use.

The best way to do this is likely to be through using MPI due to the dependency each each intermediate distribution is generated from the previous one. As a result the previous approach of using an array of parallel runs that are independent of each other will no longer work.

Let us consider the various processes that need to be run and which will run in parallel.

1. Defining the model problem for a given configuration and data set.
2. Generating parameters from a prior/intermediate distribution (done in parallel).
3. Running the models running the model enough times to generate N particles.
4. Collecting together particles in parameter space to generate intermediate/posterior distributions.
5. Writing of intermediate and posterior distributions to file.