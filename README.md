## env
- python 3.6
- cuda 11.1
- cudnn 8.5
- torch 1.7

pip install -r requirements.txt  

## preprocess
python setup.py bulid develop  
python init_predicate_matrix.py  

download pretrained model: https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl  

cd fvcore  
pip install -e fvcore  
cd ..  
python setup.py install develop  

## train
python main.py --config configs/VROID/our.yaml --mode train_relation  

## test
python main.py --config configs/VROID/test_our.yaml --mode test_relation  

## evaluate
python evaluate.py --pred_json output/test_our.yaml.json  --top_n 0  
python evaluate.py --pred_json output/test_our.yaml.json  --top_n 10  
python evaluate.py --pred_json output/test_our.yaml.json  --top_n 20  
python evaluate.py --pred_json output/test_our.yaml.json  --top_n 50  
python evaluate.py --pred_json output/test_our.yaml.json  --top_n 100  

## component analysis
./component_analysis_train.sh  
./component_analysis_test.sh  

## comparison with other methods
preparing data for detection result or training detector in baselines  
python main.py --config configs/VROID/Base-Panoptic-FPN.yaml --mode test_panoptic  

methods for comparison:  
- [STA](https://github.com/yangxuntu/vtranse.git)
- [MFULRN](https://github.com/pranoyr/visual-relationship-detection.git)
- [IMP](https://github.com/danfeiX/scene-graph-TF-release.git)
- [Graph R-CNN](https://github.com/jwyang/graph-rcnn.pytorch)
- [neural motifs](https://github.com/rowanz/neural-motifs.git)
- [VCTree](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git)
- [DSS](https://github.com/Andrew-Qibin/DSS.git)
- [NLDF](https://github.com/zhimingluo/NLDF.git)
- [ARNet](https://github.com/chenxinpeng/ARNet.git)
- [MMT](https://github.com/aimagelab/meshed-memory-transformer.git)
- [DSG](https://github.com/shikorab/DSG.git)
- Stanford CoreNLP
- Frequency: ./frequency.sh
