#!/bin/sh

source /usr/local/jobclient/config/.hive_config.sh
source /usr/local/jobclient/lib/source $0
source /usr/local/jobclient/demo/execute_modular.sh
source ~/.bash_profile

PROGRAM=$(readlink -m $0);
WORKDIR=$(dirname $(readlink -m $0));

ml-submit \
   --app-type "tensorflow" \
   --app-name "test_wide_n_deep" \
   --files city_area_code.txt,feature_column.py,input_data.py,interest_category.txt,mobile_brand.txt,creative_style_id.txt,industry_id.txt,life_status_target.txt,psid.txt,wide_n_deep.py \
   --cacheArchive viewfs://c9/user_ext/lele6/ml/pyenv/Python-tf1-13-1.zip#Python \
   --launch-cmd "Python/bin/python wide_n_deep.py --train_dir=hdfs://train/ --eval_dir=hdfs://eval --model_dir=hdfs://model_ckpt_test/ --model=wnd --learning_rate=0.001 --typ=realtime" \
   --worker-memory 32G \
   --worker-num 30 \
   --worker-cores 6 \
   --ps-memory 32G \
   --ps-num 4 \
   --ps-cores 6 \
   --tf-evaluator true \

# provide your Python file in --cacheArchive
# need chmod 777 estimatorDemoModel
