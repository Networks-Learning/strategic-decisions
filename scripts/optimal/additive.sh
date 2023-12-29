m=(4 5 6 7 8 9 100 200 300 400 500 600)
kappa=(0.1 0.25 0.5)
seed_seq=$(seq 1 100)
gamma=0.3
cost_method=uniform
for j in {0..11}
do
  for s in $seed_seq
  do
    for k in {0..2}
    do
      if [ "$j" -lt "6" ]
      then python -m lib.bruteforce --output=./outputs/optimal/additive_bruteforce_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --kappa=${kappa[$k]} --seed=$s --additive --gamma=$gamma --cost_method=${cost_method} #&
      fi
      python -m lib.thres --output=./outputs/optimal/additive_thres_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --kappa=${kappa[$k]} --seed=$s --additive --gamma=$gamma --cost_method=${cost_method} #&
      python -m lib.iterative --output=./outputs/optimal/additive_iterative_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --njobs=1 --kappa=${kappa[$k]} --seed=$s --additive --gamma=$gamma --cost_method=${cost_method} #&
      python -m lib.dp --output=./outputs/optimal/additive_dp_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --kappa=${kappa[$k]} --seed=$s --gamma=$gamma --cost_method=${cost_method} #&
    done
  done
  # wait
done