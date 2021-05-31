python main.py --config configs/VROID/test_our.yaml --mode test_relation
python main.py --config configs/VROID/test_triplet_as_output.yaml --mode test_relation
python main.py --config configs/VROID/test_only_raw_predicate.yaml --mode test_relation
python main.py --config configs/VROID/test_no_instance.yaml --mode test_relation
python main.py --config configs/VROID/test_no_semantics_features.yaml --mode test_relation
python main.py --config configs/VROID/test_no_locations_features.yaml --mode test_relation
python main.py --config configs/VROID/test_bce_loss.yaml --mode test_relation

python evaluate.py --pred_json output/test_triplet_as_output.yaml_nopair.json --top_n 0
python evaluate.py --pred_json output/test_triplet_as_output.yaml_nopair.json --top_n 10
python evaluate.py --pred_json output/test_triplet_as_output.yaml_nopair.json --top_n 20
python evaluate.py --pred_json output/test_triplet_as_output.yaml_nopair.json --top_n 50
python evaluate.py --pred_json output/test_triplet_as_output.yaml_nopair.json --top_n 100

python evaluate.py --pred_json output/test_triplet_as_output.yaml.json --top_n 0
python evaluate.py --pred_json output/test_triplet_as_output.yaml.json --top_n 10
python evaluate.py --pred_json output/test_triplet_as_output.yaml.json --top_n 20
python evaluate.py --pred_json output/test_triplet_as_output.yaml.json --top_n 50
python evaluate.py --pred_json output/test_triplet_as_output.yaml.json --top_n 100

python evaluate.py --pred_json output/test_our.yaml_nopair.json --top_n 0
python evaluate.py --pred_json output/test_our.yaml_nopair.json --top_n 10
python evaluate.py --pred_json output/test_our.yaml_nopair.json --top_n 20
python evaluate.py --pred_json output/test_our.yaml_nopair.json --top_n 50
python evaluate.py --pred_json output/test_our.yaml_nopair.json --top_n 100

python evaluate.py --pred_json output/test_only_raw_predicate.yaml.json --top_n 0
python evaluate.py --pred_json output/test_only_raw_predicate.yaml.json --top_n 10
python evaluate.py --pred_json output/test_only_raw_predicate.yaml.json --top_n 20
python evaluate.py --pred_json output/test_only_raw_predicate.yaml.json --top_n 50
python evaluate.py --pred_json output/test_only_raw_predicate.yaml.json --top_n 100

python evaluate.py --pred_json output/test_no_instance.yaml.json --top_n 0
python evaluate.py --pred_json output/test_no_instance.yaml.json --top_n 10
python evaluate.py --pred_json output/test_no_instance.yaml.json --top_n 20
python evaluate.py --pred_json output/test_no_instance.yaml.json --top_n 50
python evaluate.py --pred_json output/test_no_instance.yaml.json --top_n 100

python evaluate.py --pred_json output/test_our.yaml_instance.json --top_n 0
python evaluate.py --pred_json output/test_our.yaml_instance.json --top_n 10
python evaluate.py --pred_json output/test_our.yaml_instance.json --top_n 20
python evaluate.py --pred_json output/test_our.yaml_instance.json --top_n 50
python evaluate.py --pred_json output/test_our.yaml_instance.json --top_n 100

python evaluate.py --pred_json output/test_no_semantics_features.yaml.json --top_n 0
python evaluate.py --pred_json output/test_no_semantics_features.yaml.json --top_n 10
python evaluate.py --pred_json output/test_no_semantics_features.yaml.json --top_n 20
python evaluate.py --pred_json output/test_no_semantics_features.yaml.json --top_n 50
python evaluate.py --pred_json output/test_no_semantics_features.yaml.json --top_n 100

python evaluate.py --pred_json output/test_no_locations_features.yaml.json --top_n 0
python evaluate.py --pred_json output/test_no_locations_features.yaml.json --top_n 10
python evaluate.py --pred_json output/test_no_locations_features.yaml.json --top_n 20
python evaluate.py --pred_json output/test_no_locations_features.yaml.json --top_n 50
python evaluate.py --pred_json output/test_no_locations_features.yaml.json --top_n 100

python evaluate.py --pred_json output/test_bce_loss.yaml.json --top_n 0
python evaluate.py --pred_json output/test_bce_loss.yaml.json --top_n 10
python evaluate.py --pred_json output/test_bce_loss.yaml.json --top_n 20
python evaluate.py --pred_json output/test_bce_loss.yaml.json --top_n 50
python evaluate.py --pred_json output/test_bce_loss.yaml.json --top_n 100

python evaluate.py --pred_json output/test_our.yaml.json --top_n 0
python evaluate.py --pred_json output/test_our.yaml.json --top_n 10
python evaluate.py --pred_json output/test_our.yaml.json --top_n 20
python evaluate.py --pred_json output/test_our.yaml.json --top_n 50
python evaluate.py --pred_json output/test_our.yaml.json --top_n 100
