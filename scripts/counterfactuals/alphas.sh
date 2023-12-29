alpha=(1 2 3.3 5 10)
k=20 # fico
# k=160 # credit
njobs=1
# dataset=credit
dataset=fico
gamma=0.9746666325676951 # fico
# gamma=0.8580724846537919 # credit

cost_method=max_percentile_shift

for i in {0..4}
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/alphas_mincost_data_${dataset}_k_${k}_a_${alpha[$i]}_ --alpha=${alpha[$i]} --k=$k --njobs=$njobs --algo=mincos --cost_method=$cost_method
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/alphas_maxcover_data_${dataset}_k_${k}_a_${alpha[$i]}_ --alpha=${alpha[$i]} --k=$k --njobs=$njobs --algo=maxcov --cost_method=$cost_method
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/alphas_greedydet_data_${dataset}_k_${k}_a_${alpha[$i]}_ --alpha=${alpha[$i]} --k=$k --njobs=$njobs --algo=gd --cost_method=$cost_method
for s in {1..20}
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=./outputs/counterfactuals/alphas_greedyrand_data_${dataset}_k_${k}_a_${alpha[$i]}_s_${s}_ --seed=$s --alpha=${alpha[$i]} --k=$k --njobs=$njobs --algo=gr --cost_method=$cost_method
done
done