python main.py --config configs/VROID/our.yaml --mode train_relation
python main.py --config configs/VROID/triplet_as_output.yaml --mode train_relation
python main.py --config configs/VROID/only_raw_predicate.yaml --mode train_relation
python main.py --config configs/VROID/no_instance.yaml --mode train_relation
python main.py --config configs/VROID/no_semantics_features.yaml --mode train_relation
python main.py --config configs/VROID/no_locations_features.yaml --mode train_relation
python main.py --config configs/VROID/bce_loss.yaml --mode train_relation