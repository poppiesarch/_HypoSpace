upstream: [CTT-Pavilion/_HypoSpace: Repository for HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace)

## Document Description

The datasets have been generated and retained; experiments are uniformly conducted based on these dataset files:

```
cd causal
datasets/node03/n3_all_observations.json
datasets/node04/n4_all_observations.json

cd 3d
datasets/3d_simple_tp1.json
datasets/3d_medium_tp2.json
datasets/3d_hard_tp3.json

cd boolean
datasets/boolean_basic_2var.json
datasets/boolean_extended_2var.json
datasets/boolean_full_2var.json
```

Before use, add the API key to the configuration file `config/config_gpt4o.yaml`.

##  Running example

The corresponding file `modules/llm_interface.py` has undergone minor adjustments based on the original version.

```python
python final_causal_benchmark.py \
  --dataset "datasets/node03/n3_all_observations.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

```python
python final_3d_benchmark.py \
  --dataset "datasets/3d_simple_tp1.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

```python
python final_boolean_benchmark.py \
  --dataset "datasets/boolean_basic_2var.json" \
  --config "config/config_gpt4o.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

