#!/usr/bin

cur_dt=20200115

# 生成词表文件
function func_gen_vocab() {
	latest_dt=`cat latest_dt.txt`
	echo "the latest_dt is $latest_dt"
	hive -e "select max(dt) as latest_dt from sds_w2v_active_fansnum" > latest_dt.txt
	cur_dt=`cat latest_dt.txt`
	echo "the cur_dit is $cur_dt"
	if [ $latest_dt -ge $cur_dt ];then
        	echo "part not update"
        	return 1
	fi

	mv vocab.txt vocab.txt_bak
	rm -rf 000000_0
	hdfs dfs -get xxxxxxxx
	mv 000000_0 vocab.txt
	return 0
}


#开始训练
function func_train_job() {
	# 创建模型目录
	hdfs dfs -mkdir viewfs://c9/user_ext/ad_engine/word2vec/model_ckpt_$cur_dt
	hdfs dfs -chmod 777  viewfs://c9/user_ext/ad_engine/word2vec/model_ckpt_$cur_dt
	
	# 执行训练
	ml-submit --app-type "tensorflow" --app-name "tf-estimator-w2v" --files vocab.txt,model.py,input_data.py,main.py --cacheArchive viewfs://c9/user_ext/zeus/VR/tools_python/Python.zip\#Python --launch-cmd "Python/bin/python main.py --train_dir=viewfs://c9/user_ext/ad_engine/word2vec/location/sds_w2v_sample_followlist/dt=$cur_dt --eval_dir=viewfs://c9/user_ext/ad_engine/word2vec/eval_dir --model_dir=viewfs://c9/user_ext/ad_engine/word2vec/model_ckpt_$cur_dt" --worker-memory 30G --worker-num 30 --worker-cores 6 --ps-memory 30G --ps-num 2 --ps-cores 6 --tf-evaluator true > result.txt 2>&1
	if [[ $? -ne 0 ]]; then
		echo "train model failed"
		return 1
	else
		return 0
	fi	
}

function func_log_backup() {
	applicationId=`cat result.txt | grep 'Got new Application' | sed 's/.*Application: \(.*\)$/\1/g'`
	echo "applicationId: "$applicationId
	yarn logs -applicationId $applicationId > yarn_logs.txt
}

function func_model_save {
	hdfs dfs -put latest_dt.txt viewfs://c9/user_ext/ad_engine/word2vec/vocab_dir
	hdfs dfs -put vocab.txt viewfs://c9/user_ext/ad_engine/word2vec/vocab_dir
	hdfs dfs -chmod 777 viewfs://c9/user_ext/ad_engine/word2vec/vocab_dir/*
}

function func_job_run {
	func_gen_vocab;
	if [[ $? -ne 0 ]]; then
		echo "func_gen_vocab failed"
		exit 1
	fi

	func_train_job;
	if [[ $? -ne 0 ]]; then
		fun_log_backup;
		exit 1
	else
		func_model_save;
	fi
}

func_job_run;
