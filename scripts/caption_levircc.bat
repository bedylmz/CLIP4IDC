@echo off
:: JAVA_HOME ayarlanıyor
set "JAVA_HOME=C:\Program Files\Java\jdk1.8.0_441\jre"
set "PATH=%JAVA_HOME%\bin;%PATH%"

:: Veri yolu
set DATA_PATH=Z:\datasets\imagesRandAUG\

:: Python komutu
python C:\Users\AliCan\Desktop\clip4idc\main_task_caption.py ^
--do_train ^
--num_thread_reader=4 ^
--epochs=16 ^
--batch_size=32 ^
--n_display=32 ^
--data_path %DATA_PATH% ^
--features_path %DATA_PATH% ^
--output_dir "C:\Users\AliCan\Desktop\clip4idc\ckpts\caption" ^
--lr 1e-4 ^
--max_words 32 ^
--batch_size_val 64 ^
--datatype="levircc" ^
--coef_lr 1e-3 ^
--freeze_layer_num 0 ^
--linear_patch 2d ^
--pretrained_clip_name ViT-B/32 ^
--seed 2021 ^
--init_model "C:\Users\AliCan\Desktop\clip4idc\ckpts\retrieval\pytorch_model.bin.14"
