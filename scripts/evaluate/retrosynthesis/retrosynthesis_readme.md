# Retrosynthesis

These scripts train a regressor from a molecule to the latent space. We can then use this network and the decoder of
 the MoleculeChef to try to predict what reactants were used to create these molecules.
Even if we do not get the correct reactants we hope that the resulting product has similar properties/structure.

## Steps to run.
1. Train the regressor by running: `regress_product_to_latents.py`
2. Run `perform_retrosynthesis.py` to run the test set through this regressor and reassemble reactant bags.
3. Run the MolecularTransformer `https://github.com/pschwllr/MolecularTransformer` to predict the products associated 
with these reactant bags. You can run the Transformer with the following command (in the transformer repo):
    ```bash
    python translate.py -model <saved_transformer_model_path> \
                        -src <tokenized_reactants_file_path> \
                        -output <output_file_name> \
                        -batch_size 300 -replace_unk -max_length 400 -fast -gpu 1 -n_best 5
    
    ```
4. Run `create_retrosynthesis_plots.py` to create the plots.

