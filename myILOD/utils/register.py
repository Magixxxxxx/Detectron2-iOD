import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.register_coco import register_coco_instances
# 数据集路径
DATASET_ROOT = '/root/userfolder/data/voc2007'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2007')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2007')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'test_woDifficult.json')

# 数据集类别元数据
DATASET_CATEGORIES = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

# 数据集的子集
# PREDEFINED_SPLITS_DATASET = {
#         v : (TRAIN_PATH, os.path.join(ANN_ROOT, v)) 
#         for v in os.listdir(ANN_ROOT) if 'train' in v
#     }
# PREDEFINED_SPLITS_DATASET.update({
#         v : (VAL_PATH, os.path.join(ANN_ROOT, v)) 
#         for v in os.listdir(ANN_ROOT) if 'test' in v
#     })

PREDEFINED_SPLITS_DATASET = {
        v : (TRAIN_PATH, os.path.join(ANN_ROOT, v)) 
        for v in os.listdir(ANN_ROOT)
    }

def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for name, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():

        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **get_dataset_instances_meta())

def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k for k,_ in DATASET_CATEGORIES.items()]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [v for _,v in DATASET_CATEGORIES.items()]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def my_register():
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            get_dataset_instances_meta(),
            os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), json_file) if "://" not in json_file else json_file,
            os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), image_root),
        )
