# RAVEN-FAIR py27_env
# /home/scxhc1/AVR-PredRNet/data/datasets
# ValueError: not supported dataset_name = RAVEN-F
# main.py: error: unrecognized arguments: --num-classes 1


#python main.py --dataset-name I-RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/
      
# RAVEN_raise
python main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/raven_raise_
               
                    
# RAVEN
               
python main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/

python main.py --dataset-name RAVEN-FAIR --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/

python main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/

# 4.PGM 
# rusume
python main.py --dataset-name PGM/neutral --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
                --batch-size 256 --lr 0.001 --wd 1e-7 \
                --ckpt ckpts/ --resume /home/scxhc1/AVR-PredRNet/ckpts/PGM/neutral-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-07-ep100-seed12345/checkpoint.pth.tar
             
# PGM to run
python main.py --dataset-name PGM/neutral --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
                --batch-size 896 --lr 0.001 --wd 1e-7 \
                --ckpt ckpts/ --workers 2 --resume /home/scxhc1/AVR-PredRNet/ckpts/PGM/neutral-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-07-ep100-seed12345/checkpoint.pth.tar
                
# test_PGM
python main.py --dataset-name PGM/neutral --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
                --batch-size 896 --lr 0.001 --wd 1e-7 \
                --ckpt ckpts/test_ --workers 2

# test_PGM pre_resize
python main.py --dataset-name PGM/neutral_resized80 --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
                --batch-size 896 --wd 1e-7 \
                --ckpt ckpts/test_ --workers 2 --resume /home/scxhc1/AVR-PredRNet/ckpts/test_PGM/neutral_resized80-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-07-ep100-seed12345/checkpoint.pth.tar --lr 0.002

# preprocess_resize
python preprocess_resize.py \
--src-dir /home/scxhc1/AVR-PredRNet/data/datasets/PGM/neutral \
--dst-dir /home/scxhc1/AVR-PredRNet/data/datasets/PGM/neutral_resized80 \
--image-size 80

    
    
    
    
    
    
    
    
    
    
    
    
    
                
# Analogy(VAD)
python main.py --dataset-name Analogy/interpolation --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 3 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-7 -p 200 \
                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.0 --classifier-drop 0.1 \
                --ckpt ckpts/


# python main.py --dataset-name PGM/interpolation --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
                --image-size 80 --num-classes 1 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
                --batch-size 256 --lr 0.001 --wd 1e-7 \
                --ckpt ckpts/

# python main.py --dataset-name Analogy/interpolation --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 \
                --image-size 80 --epochs 3 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-7 -p 200 \
                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.0 --classifier-drop 0.1 \
                --ckpt ckpts/

# python main.py --dataset-name CLEVR-Matrix --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
#                --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-8 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.0 --classifier-drop 0.1 \
#                --ckpt ckpts/ --in-channels 3
