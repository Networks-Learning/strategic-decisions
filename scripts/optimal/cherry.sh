alpha=(1 2 3.3 5 10)
dataset=credit
gamma=0.8580724846537919 # credit
s=1
for i in {0..4}
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/cherry_thres_data_${dataset}_a_${alpha[$i]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=th --seed=$s
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/cherry_iterative_data_${dataset}_a_${alpha[$i]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=it --seed=$s
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/cherry_iterative_components_data_${dataset}_a_${alpha[$i]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=itc --seed=$s
done