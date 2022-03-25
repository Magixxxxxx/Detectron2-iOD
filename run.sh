# python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50boxes.json', )" OUTPUT_DIR './output/emm_15+5_boxes'

python mytrain.py --num-gpus 4 --config-file "myILOD/configs/voc.yaml" DATASETS.TRAIN "('15+5_50imgs.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_1e-2'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_18000'

python distill.py --num-gpus 4 --config-file "myILOD/configs/voc.yaml" DATASETS.TRAIN "('[1,15]_train.json', )" OUTPUT_DIR './output/base_15_False_b4'


# memory

python mytrain.py --num-gpus 8 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('emm+5.json', )" OUTPUT_DIR './output/emmix_r0.5_0.001' SOLVER.BASE_LR 0.001
python mytrain.py --num-gpus 8 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('emm15+5_50img.json', )" DATASETS.MEMORY "" OUTPUT_DIR './output/ft_0.001' SOLVER.BASE_LR 0.001


# distill

# 单卡
python distill.py --num-gpus 1 --config-file "myILOD/configs/distill.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5  SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0002 OUTPUT_DIR './output/test'

# 多卡测试
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" DATASETS.TRAIN "('[16,20]_train.json', )" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/default'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/metas_distill'

# 实验
python distill.py --num-gpus 4 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/default'


python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0002 OUTPUT_DIR './output/b2_ag_0002'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/b2_ag_0001'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0002 OUTPUT_DIR './output/b4_ag_0002'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/b4_ag_0005'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR './output/b4_ag_001'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0002 OUTPUT_DIR './output/b2_0002'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/b2_0005'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0002 OUTPUT_DIR './output/b4_0002'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/default'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill_agnostic.yaml" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR './output/b4_001'

python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" DATASETS.TRAIN "('voc_2007_trainval', )" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/b2_0005_trainval'
python distill.py --num-gpus 2 --config-file "myILOD/configs/distill.yaml" DATASETS.TRAIN "('voc_2007_train', )" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 OUTPUT_DIR './output/b2_0005_train'

# eval
python distill.py --num-gpus 4 --eval-only --config-file "myILOD/configs/distill_agnostic.yaml" MODEL.WEIGHTS "output/b4_0005/model_0003999.pth" IOD.OLD_CLS 15 IOD.NEW_CLS 5 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR './output/eval'

python distill.py --config-file "myILOD/configs/firststep.yaml"
python distill.py --resume --config-file "myILOD/configs/incre.yaml" MODEL.WEIGHTS "output/base_15/model_final.pth" OUTPUT_DIR './output/base15_+5'
python distill.py --resume --config-file "myILOD/configs/incre.yaml" MODEL.WEIGHTS "output/base15_fasterilod/model_final.pth" OUTPUT_DIR './output/base15_fasterilod_+5'