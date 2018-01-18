# BayesCMD #
BayesCMD is an updated version of the BCMD framework developed at UCL.
> BCMD is a system for defining and solving differential algebraic equation-based models, in particular for the modelling of cerebral physiology. Models are defined in a simple text language, which is then compiled into a command-line application using the RADAU5 DAE solver of Hairer & Wanner.

It has been rewritten to use Python 3.6, whilst the C and Fortran libraries and code that it uses and links to remain the same.

The intention is to developed this new framework to use both the existing optimisation and sensitivity analysis approaches from BCMD, as well as a new Approximate Bayesian Computation (ABC) method for parameter inference.

This is a working project and as such the code at present is not at a release stage. Those wishing to use BCMD are invited to use either the [original BCMD framework](https://github.com/bcmd/BCMD), if a GUI  is required, or [bcmd-docker](https://github.com/buck06191/bcmd-docker), which is currently a command line only version of BCMd, but one which provides much easier installation and better cross platform usability. 

