#!/bin/sh
ml-submit \
   --app-type "tensorflow" \
   --app-name "tf-estimator-w2v" \
   --files vocab.txt,model.py,input_data.py,main.py \
   --cacheArchive viewfs://c9/user_ext/zeus/VR/tools_python/Python.zip#Python \
   --launch-cmd "Python/bin/python main.py --train_dir=./train_dir --eval_dir=./eval_dir --model_dir=model_ckpt" \
   --worker-memory 30G \
   --worker-num 30 \
   --worker-cores 6 \
   --ps-memory 30G \
   --ps-num 2 \
   --ps-cores 6 \
   --tf-evaluator true

