alpha=3.3
dataset=credit
gamma=0.8563985631883158 # credit
seed_seq=$(seq 1 100)
confounding_seq=$(seq 0.0 0.05 1.0)
cost_method=max_percentile_shift
for confounding in $confounding_seq
do
    for s in $seed_seq
    do
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/confounding_thres_data_${dataset}_cost_${cost_method}_level_${confounding}_seed_${s} --alpha=$alpha --njobs=1 --algo=th --seed=$s --cost_method=$cost_method --confounding=$confounding #&
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/confounding_iterative_data_${dataset}_cost_${cost_method}_level_${confounding}_seed_${s} --alpha=$alpha --njobs=1 --algo=itc --seed=$s --cost_method=$cost_method --confounding=$confounding #&
    done
    #wait
done