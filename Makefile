.PHONY: test baseline institutional external-data data-engineer ml notebooks

test:
	python -m unittest -v test_delta1_cta.py test_institutional_strategy.py test_ml_strategy.py

baseline:
	python delta1_cta.py --data-dir "$(DELTA1_DATA_DIR)" --output-dir outputs

institutional:
	python institutional_strategy.py --data-dir "$(DELTA1_DATA_DIR)" --output-dir outputs

external-data:
	python download_external_data.py

data-engineer:
	python data_engineer_features.py --input "$(DELTA1_DATA_ENGINEER_INPUT)" --output-dir outputs

ml:
	python ml_strategy.py --data-dir "$(DELTA1_DATA_DIR)" --external-macro data/external/fred_macro.csv --output-dir outputs

notebooks:
	jupyter nbconvert --to notebook --execute --inplace DELTA1_CTA_Strategy.ipynb
	jupyter nbconvert --to notebook --execute --inplace INSTITUTIONAL_STRATEGY.ipynb
	jupyter nbconvert --to notebook --execute --inplace ML_WALK_FORWARD_STRATEGY.ipynb
