## environment
- python 3.8
- cuda 11.1
- cudnn 8.0
- torch 1.7
apt-get install libglib2.0-dev libsm6 libxrender-dev libxext6  
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html  
pip install pyyaml==5.4.1 --ignore-installed  
pip install -r requirements.txt  
 
or:  
docker pull yfraquelle/vroid_env:v1  
docker run -it -v /host/path/to/VROID:/docker/path/to/VROID --ipc=host --net=host <image_id> /bin/bash  

## dataset
We constructed a ViROI dataset for VROID based on the 45,000 images in the IOID dataset and their corresponding captions in the MSCOCO dataset. After filtering the images without VROIs, the ViROI dataset contains 30,120 images. It is further divided into the training set (25,091 images with 91,496 VROIs) and the test set (5,029 images with 18,268 VROIs).  
Download dataset to  
../data  
├── [viroi_json](https://drive.google.com/file/d/1PwntYlHar803vArwLV9Ba2KaRl9BT7ee/view?usp=sharing)  
│   ├── train_images_dict.json  
│   ├── train_images_triplets_dict.json  
│   ├── test_images_dict.json  
│   ├── test_images_triplets_dict.json  
│   ├── class_dict.json  
│   └── relation_dict.json  
├── [ioid_images](https://drive.google.com/file/d/1yRyduTD58_lL1GI4oGoUdhpi3gnjzvgO/view?usp=sharing) (MSCOCO images filtered by IOID)  
├── [ioid_panoptic](https://drive.google.com/file/d/1nxvSLhNkk7Vc2HEEXquG51tESwEHK07T/view?usp=sharing) (MSCOCO panoptic annotation images filtered by IOID)  
└── viroi_stuff (python prepare_panoptic.py)  

## preprocess
python setup.py bulid develop    
python init_predicate_matrix.py  

download pretrained model: https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl  

python setup.py install develop  

## train
python main.py --config configs/VROID/our.yaml --mode train_relation  
The pretrained model weights can be [downloaded](https://drive.google.com/file/d/1-QOTkAUbfFzilNWHoXL6BOonOMtmxjML/view?usp=sharing).  

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
The [pretrained model weights](https://1drv.ms/u/s!AqIJSYD5gt-YjV1jEVu0nMn3b0Ym?e=jfic91) can be downloaded and unzip to the output folder.  

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
