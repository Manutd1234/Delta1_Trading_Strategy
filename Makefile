.PHONY: test baseline institutional notebooks

test:
	python -m unittest -v test_delta1_cta.py test_institutional_strategy.py

baseline:
	python delta1_cta.py --data-dir "$(DELTA1_DATA_DIR)" --output-dir outputs

institutional:
	python institutional_strategy.py --data-dir "$(DELTA1_DATA_DIR)" --output-dir outputs

notebooks:
	jupyter nbconvert --to notebook --execute --inplace DELTA1_CTA_Strategy.ipynb
	jupyter nbconvert --to notebook --execute --inplace INSTITUTIONAL_STRATEGY.ipynb
