""" 
Loads the results of the control.py file from pickle and plots them.

We visualise the final trajectory and controls as well as the convergence
behaviour of the NLP and the sparsity of the jacobian.

"""
import os, sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from src.plotting import plot_trajectory

def main():
    pass

if __name__ == "__main__":
    main()