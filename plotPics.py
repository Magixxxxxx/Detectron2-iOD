

import os,sys
import numpy as np
from setuptools import setup
import torch, torchvision
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data import detection_utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader

def getModel(cfg, params):
    model = build_model(cfg)
    model.eval()
    sd = torch.load(params)['model']  # why map_location leads to cruption
    model.load_state_dict(sd, strict = False)
    return model

def drawAndSave(img, metadata, pred, output_name):
    visualizer = Visualizer(img, metadata=metadata, scale=1.0)
    vis = visualizer.draw_instance_predictions(pred['instances'])

    def output(vis, fname):
        os.makedirs("plotPics", exist_ok=True)
        filepath = os.path.join("plotPics", fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)
    output(vis, output_name)

def plotPred():

    cfg = get_cfg()
    cfg.merge_from_file("myILOD/configs/voc.yaml")

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    params_new = "output/distill_10img_[f_box_roi]_15+5_0.1d_mix[on_0.7]/model_final.pth"
    params_old = "output/base_15/model_final.pth" 
    cur = 5
    pre = 15
    model_new = getModel(cfg, params_new)
    model_old = getModel(cfg, params_old)
    dataloader = build_detection_test_loader(cfg, 'voc_2007_test')

    for dic in dataloader:
        dic = dic[0]
        img = detection_utils.read_image(dic["file_name"], "RGB")
        imgs = [{"image": torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))}]
        with torch.no_grad(): pred_new = model_new(imgs)[0]

        mask = pred_new['instances'].scores >= 0.7
        pred_new['instances'].pred_boxes.tensor = pred_new['instances'].pred_boxes.tensor[mask]
        pred_new['instances'].scores = pred_new['instances'].scores[mask]
        pred_new['instances'].pred_classes = pred_new['instances'].pred_classes[mask]

        for cls in pred_new['instances'].pred_classes:
            if cls in range(pre+1, pre + cur+1):
                with torch.no_grad(): pred_old = model_old(imgs)[0]
                mask = pred_old['instances'].scores >= 0.7

                pred_old['instances'].pred_boxes.tensor = pred_old['instances'].pred_boxes.tensor[mask]
                pred_old['instances'].scores = pred_old['instances'].scores[mask]
                pred_old['instances'].pred_classes = pred_old['instances'].pred_classes[mask]

                drawAndSave(img, metadata, pred_new, os.path.basename(dic["file_name"] + '.jpg'))
                drawAndSave(img, metadata, pred_old, os.path.basename(dic["file_name"]))
            else:
                break

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("myILOD/configs/voc.yaml")
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    lambd = np.random.beta(5,5)
    img1 = torch.tensor(detection_utils.read_image("000366.jpg","RGB")) 
    img2 = torch.tensor(detection_utils.read_image("000384.jpg","RGB"))

    # operation
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])
    
    mix_img = torch.zeros([height, width, 3])
    mix_img[:img1.shape[0], :img1.shape[1], :] = img1 * lambd
    mix_img[:img2.shape[0], :img2.shape[1], :] += img2 * (1. - lambd)

    visualizer = Visualizer(mix_img, metadata=metadata, scale=1.0)
    vis = visualizer.draw_box([10,185,110,405],edge_color="g")
    vis = visualizer.draw_box([240,205,370,425],edge_color="g")
    vis = visualizer.draw_text("chair", [10 + 33,185 - 30], color="g")
    vis = visualizer.draw_text("chair", [240 + 33,205 - 30], color="g")

    vis = visualizer.draw_box([169,23,453,291],edge_color="b")
    vis = visualizer.draw_text("sofa", [169 + 34,25], color="b")

    def output(vis, fname):
        os.makedirs("plotPics", exist_ok=True)
        filepath = os.path.join("plotPics", fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

    output(vis, "0.jpg")
    

# batched_input = {}
# batched_input['image'] = torch.tensor(detection_utils.convert_PIL_to_numpy(img, "BGR")) 
# print(model.inference([batched_input]))
# a = ImageDraw.ImageDraw(img)
# # for b in each_img['instances']._fields['gt_boxes'].tensor:
# #     a.rectangle([int(i) for i in b])
# img.save("0.jpg")