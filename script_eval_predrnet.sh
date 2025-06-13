# /home/scxhc1/AVR-PredRNet/data/datasets
# /home/scxhc1/AVR-PredRNet/ckpts/RAVEN-FAIR-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed12345/model_best.pth.tar
# /home/scxhc1/AVR-PredRNet/ckpts/Analogy/interpolation-predrnet_analogy-prb3-b0.0c0.1-imsz80-wd1e-07-ep3-seed12345/model_best.pth.tar

python main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume /home/scxhc1/AVR-PredRNet/ckpts/RAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed12345/model_best.pth.tar \
               --show-detail

python main.py --dataset-name RAVEN-FAIR --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume /home/scxhc1/AVR-PredRNet/ckpts/RAVEN-FAIR-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed12345/model_best.pth.tar \
               --show-detail
               
python main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume /home/scxhc1/AVR-PredRNet/ckpts/I-RAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed12345/model_best.pth.tar \
               --show-detail
               
               
python main.py --dataset-name Analogy --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1 \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume /home/scxhc1/AVR-PredRNet/ckpts/Analogy/interpolation-predrnet_analogy-prb3-b0.0c0.1-imsz80-wd1e-07-ep3-seed12345/model_best.pth.tar \
               --show-detail
               


python main.py --dataset-name PGM/neutral_resized80 --dataset-dir /home/scxhc1/AVR-PredRNet/data/datasets --gpu 0,1  \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume /home/scxhc1/AVR-PredRNet/ckpts/test_PGM/neutral_resized80-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-07-ep100-seed12345/model_best.pth.tar \
               --show-detail

