#!/bin/bash

# Execute each Jupyter Notebook using nbconvert without a timeout
jupyter nbconvert --to notebook --execute "CNN_VT.ipynb"
jupyter nbconvert --to notebook --execute "CNN_spec.ipynb"
jupyter nbconvert --to notebook --execute "MLP_spec.ipynb"
jupyter nbconvert --to notebook --execute "MLP_VT.ipynb"

echo "All notebooks have been executed."
