#!/bin/bash

echo TemporalKNN -----------------------------------------

virtualenv venv

source venv/bin/activate

pip install scikit-surprise==1.0.2

python temporal_algo.py

deactivate

echo prepare librec data ---------------------------------

sed 's/::/ /g' ~/.surprise_data/ml-1m/ml-1m/ratings.dat > /tmp/ratings.dat

echo librec 1.3 TimeSVD++ --------------------------------

java -jar librec-1.3/librec.jar -c librec-timeSVD++.conf

echo librec 1.3 SVD++ ------------------------------------

java -jar librec-1.3/librec.jar -c librec-SVD++.conf

echo librec Result Format: algorithm name, performance_MAE, RMSE, NMAE, rMAE, rRMSE, MPE_, recommender options _learning rate, reg.user, reg.item, num.factors, num.max.iter, bold.driver_, execution time _training time and testing time_
