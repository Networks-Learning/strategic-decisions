alpha=2
# k=(16 32 80 160) # credit (3200)
k=(2 4 10 20 40 100) # fico (400)
njobs=1
# dataset=credit
dataset=fico
gamma=0.9746666325676951 # fico
# gamma=0.8580724846537919 # credit

cost_method=max_percentile_shift

# for i in {0..3} # credit 
for i in {0..5} # fico (leakage)
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/real_mincost_data_${dataset}_k_${k[$i]}_a_${alpha}_ --alpha=$alpha --k=${k[$i]} --njobs=$njobs --algo=mincos --cost_method=$cost_method
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/real_maxcover_data_${dataset}_k_${k[$i]}_a_${alpha}_ --alpha=$alpha --k=${k[$i]} --njobs=$njobs --algo=maxcov --cost_method=$cost_method
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=outputs/counterfactuals/real_greedydet_data_${dataset}_k_${k[$i]}_a_${alpha}_ --alpha=$alpha --k=${k[$i]} --njobs=$njobs --algo=gd --cost_method=$cost_method
for s in {1..20}
do
python -m lib.real  --data=data/processed/$dataset --gamma=$gamma --output=./outputs/counterfactuals/real_greedyrand_data_${dataset}_k_${k[$i]}_a_${alpha}_s_${s}_ --seed=$s --alpha=$alpha --k=${k[$i]} --leaking=[0.1,0.5,0.9] --njobs=$njobs --algo=gr --cost_method=$cost_method
done
done