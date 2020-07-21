
echo "start predict_split_0000000"
python3 predict.py ../data/csv/predict_split_0000000  presult.txt
wait

echo "start predict_split_0000001"
python3 predict.py ../data/csv/predict_split_0000001  presult.txt
wait

echo "start predict_split_0000002"
python3 predict.py ../data/csv/predict_split_0000002  presult.txt

