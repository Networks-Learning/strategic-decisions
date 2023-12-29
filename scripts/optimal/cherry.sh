alpha=(1 2 3.3 5 10)
dataset=credit
gamma=(0.8326266551572056 0.8563985631883158 0.8719260871092159) # credit
s=1
cost_method=max_percentile_shift
for i in {0..4}
do
    for g in {0..2}
    do
        python -m lib.real  --data=data/processed/$dataset --gamma=${gamma[$g]} --output=outputs/optimal/cherry_thres_data_${dataset}_cost_${cost_method}_a_${alpha[$i]}_gamma_${gamma[$g]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=th --seed=$s --cost_method=$cost_method
        python -m lib.real  --data=data/processed/$dataset --gamma=${gamma[$g]} --output=outputs/optimal/cherry_iterative_data_${dataset}_cost_${cost_method}_a_${alpha[$i]}_gamma_${gamma[$g]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=it --seed=$s --cost_method=$cost_method
        python -m lib.real  --data=data/processed/$dataset --gamma=${gamma[$g]} --output=outputs/optimal/cherry_iterative_components_data_${dataset}_cost_${cost_method}_a_${alpha[$i]}_gamma_${gamma[$g]}_seed_${s} --alpha=${alpha[$i]} --njobs=1 --algo=itc --seed=$s --cost_method=$cost_method
    done
done