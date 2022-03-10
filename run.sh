# python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50boxes.json', )" OUTPUT_DIR './output/emm_15+5_boxes'
python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_1e-2'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_18000'
python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_b8'


# memory

python mytrain.py --num-gpus 8 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('emm+5.json', )" OUTPUT_DIR './output/emmix_r0.5_0.001' SOLVER.BASE_LR 0.001
python mytrain.py --num-gpus 8 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('emm15+5_50img.json', )" DATASETS.MEMORY "" OUTPUT_DIR './output/ft_0.001' SOLVER.BASE_LR 0.001


# distill
python distill.py --num-gpus 1 --config-file "myILOD/configs/distill.yaml" DATASETS.TRAIN "('[16,20]_train.json', )" SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR './output/test'

python myILOD/utils/get_inter_output.py --num-gpus 1 --config-file "myILOD/configs/distill.yaml" DATASETS.TRAIN "('[16,20]_train.json', )" SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR './output/test'

python distill.py --num-gpus 4 --config-file "myILOD/configs/distill.yaml" DATASETS.TRAIN "('[16,20]_train.json', )" SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR './output/no_distill'

