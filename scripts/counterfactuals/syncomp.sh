m=(10 20 50 100 200)
sparsity=(5 10 25 50 100)
k=(1 2 5 10 20)

kp=(1 2 5 10 20 30 40)
for s in {1..20}
do
  for i in {0..3}
  do
  python -m lib.min_cost --output=./outputs/counterfactuals/syncomp_mincost_m_${m[$i]}_k_${k[$i]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${k[$i]} --sparsity=${sparsity[$i]} --seed=${s}
  python -m lib.max_cover --output=./outputs/counterfactuals/syncomp_maxcover_m_${m[$i]}_k_${k[$i]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${k[$i]} --sparsity=${sparsity[$i]} --seed=${s}
  python -m lib.greedy_deter --output=./outputs/counterfactuals/syncomp_greedydet_m_${m[$i]}_k_${k[$i]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${k[$i]} --sparsity=${sparsity[$i]} --seed=${s}
  python -m lib.greedy_rand --output=./outputs/counterfactuals/syncomp_greedyrand_m_${m[$i]}_k_${k[$i]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${k[$i]} --sparsity=${sparsity[$i]} --seed=${s}
  done
  i=4
  for j in {0..6}
  do
    python -m lib.min_cost --output=./outputs/counterfactuals/syncomp_mincost_m_${m[$i]}_k_${kp[$j]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${kp[$j]} --sparsity=${sparsity[$i]} --seed=${s}
    python -m lib.max_cover --output=./outputs/counterfactuals/syncomp_maxcover_m_${m[$i]}_k_${kp[$j]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${kp[$j]} --sparsity=${sparsity[$i]} --seed=${s}
    python -m lib.greedy_deter --output=./outputs/counterfactuals/syncomp_greedydet_m_${m[$i]}_k_${kp[$j]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${kp[$j]} --sparsity=${sparsity[$i]} --seed=${s}
    python -m lib.greedy_rand --output=./outputs/counterfactuals/syncomp_greedyrand_m_${m[$i]}_k_${kp[$j]}_sparsity_${sparsity[$i]}_seed_${s}_optimal.csv --m=${m[$i]} --k=${kp[$j]} --sparsity=${sparsity[$i]} --seed=${s}
  done
done
