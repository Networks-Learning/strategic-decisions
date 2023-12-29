alpha=3.3
dataset=credit
gamma=0.8563985631883158 # credit
seed_seq=$(seq 1 100)
beta_seq=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)
cost_method=max_percentile_shift
for i in {0..9}
do
    for s in $seed_seq
    do
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/noisy_thres_data_${dataset}_cost_${cost_method}_noisy_pyx_beta_${beta_seq[$i]}_seed_${s} --alpha=$alpha --njobs=1 --algo=th --seed=$s --cost_method=$cost_method --noisy_pyx=${beta_seq[$i]} #&
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/noisy_iterative_data_${dataset}_cost_${cost_method}_noisy_pyx_beta_${beta_seq[$i]}_seed_${s} --alpha=$alpha --njobs=1 --algo=itc --seed=$s --cost_method=$cost_method --noisy_pyx=${beta_seq[$i]} #&
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/noisy_thres_data_${dataset}_cost_${cost_method}_noisy_cost_beta_${beta_seq[$i]}_seed_${s} --alpha=$alpha --njobs=1 --algo=th --seed=$s --cost_method=$cost_method --noisy_cost=${beta_seq[$i]} #&
        python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/optimal/noisy_iterative_data_${dataset}_cost_${cost_method}_noisy_cost_beta_${beta_seq[$i]}_seed_${s} --alpha=$alpha --njobs=1 --algo=itc --seed=$s --cost_method=$cost_method --noisy_cost=${beta_seq[$i]} #&
    done
    #wait
done