m=(4 5 6 7 8 9 100 200 300 400 500 600)
kappa=(0.25 0.5 0.75)
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
      then python -m lib.bruteforce --output=./outputs/optimal/khayyam_bruteforce_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --seed=$s --gamma=$gamma --kappa=${kappa[$k]} --cost_method=${cost_method} #&
      fi
      python -m lib.thres --output=./outputs/optimal/khayyam_thres_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --seed=$s --gamma=$gamma --kappa=${kappa[$k]} --cost_method=${cost_method} #&
      python -m lib.iterative --output=./outputs/optimal/khayyam_iterative_cost_${cost_method}_gamma_${gamma}_kappa_${kappa[$k]}_m_${m[$j]}_seed_$s --m=${m[$j]} --seed=$s --gamma=$gamma --njobs=1 --kappa=${kappa[$k]} --cost_method=${cost_method} #&
    done
  done
  # wait
done

