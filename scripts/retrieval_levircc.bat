@echo off
set DATA_PATH=Z:\datasets\imagesRandAUG\
python "C:\Users\AliCan\Desktop\clip4idc\main_task_retrieval.py" ^
--do_train ^
--num_thread_reader 4 ^
--epochs 32 ^
--batch_size 32 ^
--n_display 32 ^
--data_path %DATA_PATH% ^
--features_path %DATA_PATH% ^
--output_dir "C:\Users\AliCan\Desktop\clip4idc\ckpts\retrieval" ^
--lr 1e-4 ^
--max_words 64 ^
--batch_size_val 32 ^
--datatype "levircc" ^
--coef_lr 1e-3 ^
--freeze_layer_num 0 ^
--linear_patch "2d" ^
--pretrained_clip_name "ViT-B/32" ^
--init_model "C:\Users\AliCan\Desktop\clip4idc\ckpts\retrieval\pytorch_model.bin.3"