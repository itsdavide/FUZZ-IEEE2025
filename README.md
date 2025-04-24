# FUZZ-IEEE2025
Approximation code for the paper: 

S. Lorenzini, A. Cinfrignini, D. Petturiti, and B. Vantaggi.
_Quantile-constrained Choquet-Wasserstein p-box  approximation of arbitrary belief functions_. 
In **Proceedings of FUZZ-IEEE 2025**.

# Requirements
Reference to the library **pyomo** is here: http://www.pyomo.org/

The code necessitates of the **bonmin** solver that can be downloaded here: https://www.coin-or.org/Bonmin/

The **bonmin** solver should be located in a folder and the path to that folder should be inserted in the variable **optimizer_path** in the top of file **projection_KL.py**

# File inventory

**projection_KL.py**

**constrained_VaR.py**
