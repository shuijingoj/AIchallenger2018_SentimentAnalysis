#!/bin/bash
for model_index in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
 python main_train.py -mn ${model_index}
done

for validation_index in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
 python main_train.py -vn ${validation_index}
done 