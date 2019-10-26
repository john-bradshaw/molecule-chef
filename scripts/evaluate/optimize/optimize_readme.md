# Optimize 
This directory contains the scripts to run the local optimization.

Steps to run:
1. run run_local_latent_search.py
2. translate the resultant tokenized reactants through the Molecular Transformer eg by:
```bash
 python translate.py -model <transformer-weight-path>> \
                    -src <path-to-tokenized-reactants> \
                    -output <path-for-tokenized-products>  \
                    -batch_size 300 -replace_unk -max_length 500 -fast -gpu 1 -n_best 5
```
3. Run `plot_results.py`.