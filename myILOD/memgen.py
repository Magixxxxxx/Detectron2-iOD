import os, json, random, copy
from collections import defaultdict
from typing import Counter

VOC_PATH = "../data/voc2007/annotations"

def getmm_img(total_d, base_cats):

    new_annotations = []

    # 预处理
    cat2img_ids = [set() for _ in range(21)] #每类的图片id
    img_id2anns = defaultdict(list) #每张图的标注

    for ann in total_d['annotations']:
        cat = ann['category_id']
        cat2img_ids[cat].add(ann['image_id'])
        img_id2anns[ann['image_id']].append(ann)
    
    # 对每类取50图
    img_id_set = set()
    for i in base_cats:
        sampled_ids = random.sample(cat2img_ids[i], 10)
        for id in sampled_ids:
            for ann in img_id2anns[id]:
                if ann['category_id'] == i: 
                    new_annotations.append(ann)
            img_id_set.add(id)
    print(len(img_id_set))
    return new_annotations

def getmm_ann(total_d, base_cats):

    new_annotations = []

    # 预处理
    cat2img_ids = [set() for _ in range(21)] #每类的图片id
    img_id2anns = defaultdict(list) #每张图的标注

    for ann in total_d['annotations']:
        cat = ann['category_id']
        cat2img_ids[cat].add(ann['image_id'])
        img_id2anns[ann['image_id']].append(ann)
    
    # 对每类取50 instances
    img_id_set = set()
    for i in base_cats:
        cur_catnums = 0
        sampled_ids = random.sample(cat2img_ids[i], 50)
        # 加够50个为止
        for id in sampled_ids:
            for ann in img_id2anns[id]:
                if ann['category_id'] == i: 
                    cur_catnums += 1
                    new_annotations.append(ann)
                img_id_set.add(id) 
            if cur_catnums >= 50: break 

    return new_annotations

def getNew(total_d, new_cats):

    new_annotations = []
    
    cat2img_ids = [set() for _ in range(21)] #每类的图片id
    img_id2anns = defaultdict(list) #每张图的标注

    for ann in total_d['annotations']:
        cat = ann['category_id']
        cat2img_ids[cat].add(ann['image_id'])
        img_id2anns[ann['image_id']].append(ann)
    
    img_id_set = set()
    for cat, ids in enumerate(cat2img_ids):
        cur_catnums = 0
        if cat in new_cats:
            for id in ids:
                for ann in img_id2anns[id]:
                    if ann['category_id'] == cat: 
                        cur_catnums += 1
                        new_annotations.append(ann)
                    img_id_set.add(id) 

    return new_annotations

def getMemory(base_cats):
    nums = base_cats[-1] - base_cats[0]
    output_filename = '{}_10img.json'.format(nums+1)
    with open(os.path.join(VOC_PATH, 'voc_train2007.json')) as total_f:
        total_d = dict(json.load(total_f))
        print(len(total_d['images']))
        print(len(total_d['annotations']))
        annotations = getmm_img(total_d, base_cats)
        # 数据信息
        meta = [0] * 21
        for ann in annotations:
            meta[ann['category_id']] += 1
        print(meta)

        final_d = copy.deepcopy(total_d)
        final_d['description'] = meta
        final_d['annotations'] = annotations
        with open(output_filename,'w') as out_f:
            out_f.write(json.dumps(final_d))
            
if __name__ == '__main__':
    a = [range(1, 11), range(1, 16), range(1, 20)]
    for r in a:
        getMemory(r)
    print("----end----")