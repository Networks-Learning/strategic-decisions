alpha=(1 2 3.3 5 10)
njobs=4
# dataset=credit
dataset=fico
gamma=0.9746666325676951 # fico
# gamma=0.8580724846537919 # credit
max_iter=20
for i in {0..4}
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/cherry_heur_data_${dataset}_a_${alpha[$i]} --alpha=${alpha[$i]} --njobs=$njobs --algo=gh
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/cherry_thres_data_${dataset}_a_${alpha[$i]} --alpha=${alpha[$i]} --algo=th
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/cherry_iterative_data_${dataset}_a_${alpha[$i]} --alpha=${alpha[$i]} --njobs=1 --algo=it
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/cherry_iterative_parallel_data_${dataset}_a_${alpha[$i]} --alpha=${alpha[$i]} --njobs=$njobs --algo=it --max_iter=$max_iter
done