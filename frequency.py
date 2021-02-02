import json
import numpy as np

def analyse_reoid():
    class_dict=json.load(open("../data/viroi_json/class_dict.json",'r'))
    relation_dict=json.load(open("../data/viroi_json/relation_dict.json",'r'))

    train_images_dict=json.load(open("../data/viroi_json/train_images_dict.json",'r'))
    train_images_triplets_dict=json.load(open("../data/viroi_json/train_images_triplets_dict.json",'r'))
    # val_images_dict.update(json.load(open("../data/viroi_json/train_images_dict.json",'r')))
    # val_images_triplets_dict.update(json.load(open("../data/viroi_json/train_images_triplets_dict.json",'r')))
    # 'person': {'on': {'chair':30,'boat':10}}
    head_dict={}
    tail_dict={}
    for subject_id in class_dict:
        head_dict[int(subject_id)]={}
        tail_dict[int(subject_id)]={}
        for relation_id in relation_dict:
            head_dict[int(subject_id)][int(relation_id)]={}
            tail_dict[int(subject_id)][int(relation_id)] = {}
            for object_id in class_dict:
                head_dict[int(subject_id)][int(relation_id)][int(object_id)] = 0
                tail_dict[int(subject_id)][int(relation_id)][int(object_id)] = 0
    # print(head_dict)
    # print(tail_dict)

    for image_id in train_images_dict:
        instances=train_images_dict[image_id]['instances']
        triplets=train_images_triplets_dict[image_id]['triplets']
        for triplet_id in triplets:
            subject_id=triplets[triplet_id]['subject_instance_id']
            object_id=triplets[triplet_id]['object_instance_id']
            relation_id=triplets[triplet_id]['relation_id']
            subject_class_id = instances[str(subject_id)]['class_id']
            object_class_id = instances[str(object_id)]['class_id']
            head_dict[subject_class_id][relation_id][object_class_id]+=1
            tail_dict[object_class_id][relation_id][subject_class_id] += 1
    # json.dump(head_dict,open("reoid_frequency.json",'w'))
    count_list=[]
    for sub_class_id in head_dict:
        for relation_class_id in head_dict[sub_class_id]:
            for obj_class_id in head_dict[sub_class_id][relation_class_id]:
                if head_dict[sub_class_id][relation_class_id][obj_class_id]>0:
                    count_list.append([class_dict[str(sub_class_id)]['name'],relation_dict[str(relation_class_id)]['name'],class_dict[str(obj_class_id)]['name'],head_dict[sub_class_id][relation_class_id][obj_class_id]])

    sorted_count_list = sorted(count_list,key=lambda x:x[3],reverse=True)

    sumnum = 0
    for gr in sorted_count_list:
        sumnum += gr[3]

    cum=0
    for i,gr in enumerate(sorted_count_list):
        cum+=gr[3]
        if i <= 0.5*len(sorted_count_list):
            print(str(gr)+" %0.4f"%(gr[3]/sumnum)+" %.4f"%(cum/sumnum))
    return sumnum, head_dict

def frequency_baseline(sumnum, reoid_frequency):
    class_dict=json.load(open("../data/viroi_json/class_dict.json",'r'))
    detectron2_dict=json.load(open("../data/viroi_json/detectron2_viroi_test_images_dict.json",'r'))

    prediction_json={}
    image_count=0
    for image_id in detectron2_dict:
        image_count+=1
        print(str(image_count)+"/"+str(len(detectron2_dict)))
        instances=detectron2_dict[image_id]['instances']
        if len(instances)>0:
            pred_classes=np.array([instances[instance_id]['class_id'] for instance_id in instances])
            pred_boxes=np.array([instances[instance_id]['bbox'] for instance_id in instances])
            single_result=[]
            single_index = []
            for i in range(len(pred_classes)):
                sub_result=[]
                for j in range(len(pred_classes)):
                    obj_result=[]
                    for k in range(249):
                        obj_result.append(reoid_frequency[pred_classes[i]][k+1][pred_classes[j]]/sumnum)
                        single_index.append([i, j, k])
                    sub_result.append(obj_result)
                single_result.append(sub_result)
            single_index = np.array(single_index)
            single_result = np.array(single_result).reshape(-1)

            single_result_indx = np.argsort(single_result)[::-1][:100]
            locations = single_index[single_result_indx]
            # print(locations)
            scores = single_result[single_result_indx]
            prediction_json[image_id] = {
                "relation_ids": (locations[:, 2] + 1).tolist(),
                "subject_class_ids": pred_classes[locations[:, 0]].tolist(),
                "subject_boxes": pred_boxes[locations[:, 0]].tolist(),
                "object_class_ids": pred_classes[locations[:, 1]].tolist(),
                "object_boxes": pred_boxes[locations[:, 1]].tolist(),
                "scores": scores.tolist()
            }
        else:
            prediction_json[image_id] = {
                "relation_ids": [],
                "subject_class_ids": [],
                "subject_boxes": [],
                "object_class_ids": [],
                "object_boxes": [],
                "scores": []
            }
    json.dump(prediction_json,open("./output/frequency_final.json",'w'))

sumnum, head_dict=analyse_reoid()
frequency_baseline(sumnum, head_dict)