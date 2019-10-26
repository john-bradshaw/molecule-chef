
# This MAKEFILE can be used inside the Docker container to run the various experiments
# It implements the steps documented in the respective readme's

py_4_mchef = /opt/conda/envs/py36_py121/bin/python
py_4_mtransformer = /opt/conda/envs/py36_py041/bin/python
mtransformer_dir = /molecular_transformer
current_dir := /molecule_chef/scripts/evaluate

all:
	@echo "Call make gen, make opt or make retro to run the respective experiments."
.PHONY: all


# Generation
gen: gen-assess-metrics
	@echo "Generation completed!"
.PHONY: gen

gen-assess-metrics: generation/generated_smiles/mchef-samples.assembled-product.txt
	@echo "Assessing metrics..."
	cd generation/metrics ; ${py_4_mchef} evaluate_metrics.py --config=tables_spec.example.json
.PHONY: gen-assess-metrics

generation/generated_smiles/mchef-samples.assembled-product.txt: generation/generate_for_mchef/mchef-samples.assembled-product.txt
	@echo Moving the generated SMILES.
	mv generation/generate_for_mchef/mchef-samples.assembled-product.txt generation/generated_smiles/mchef-samples.assembled-product.txt

generation/generate_for_mchef/mchef-samples.tokenized-reactant.txt:
	@echo "Creating the reactant bag samples..."
	cd generation/generate_for_mchef ; ${py_4_mchef}  create_reactant_bags.py ../../../train/molecule_chef_no_prop_weights_last_epoch.pth.pick \
	    mchef-samples.tokenized-reactant.txt


# Optimize
opt: optimize/kde_results.pdf
	@echo "Optimization completed!"
.PHONY: opt

optimize/kde_results.pdf: optimize/opt.tokenized-reactant.txt optimize/opt.tokenized-product.txt
	@echo Creating plots
	cd optimize ; ${py_4_mchef} plot_results.py opt.tokenized-reactant.txt opt.tokenized-product.txt --nbest=1

optimize/opt.tokenized-reactant.txt:
	@echo Running the local search
	cd optimize ; ${py_4_mchef} run_local_latent_search.py ../../train/molecule_chef_weights_last_epoch.pth.pick


# Retrosynthesis
retro: retro-create-plots
	@echo "Retrosynthesis completed!"
.PHONY: retro

retro-create-plots: retrosynthesis/op/retro.tokenized-reactant.txt retrosynthesis/op/retro.tokenized-product.txt
	cd retrosynthesis ; \
	 ${py_4_mchef} create_retrosynthesis_plots.py op/retro.tokenized-reactant.txt op/retro.tokenized-product.txt --nbest=1
.PHONY: retro-create-plots

retrosynthesis/op/retro.tokenized-reactant.txt: retrosynthesis/chkpts/final-weights.pth.pick
	cd retrosynthesis ; \
	${py_4_mchef} perform_retrosynthesis.py ../../train/molecule_chef_weights_last_epoch.pth.pick chkpts/final-weights.pth.pick

retrosynthesis/chkpts/final-weights.pth.pick:
	cd retrosynthesis ; \
	${py_4_mchef} regress_product_to_latents.py ../../train/molecule_chef_weights_last_epoch.pth.pick


# Predict Reactions (used in the methods above)
%.tokenized-product.txt: %.tokenized-reactant.txt
	@echo "Running transformer on $<"
	cd ${mtransformer_dir} ;  ${py_4_mtransformer} translate.py -model molecular_transformer_weights.pt \
	                    -src ${current_dir}/$< \
	                    -output ${current_dir}/$@  \
	                    -batch_size 300 -replace_unk -max_length 500 -fast -gpu 1 -n_best 1


%.assembled-product.txt: %.tokenized-product.txt
	@echo "Reassembling (from tokenized) $<"
	${py_4_mchef} put_together_molecular_transformer_predictions.py $< $@ --nbest=1