m=(4 5 6 7 8 9 100 200 300 400 500 600)
kappa=(0.1 0.25 0.5)
seed=(1 2 3 4 5 6 7 8 9 10)
gamma=0.3
for i in {0..9}
do
  for j in {0..11}
  do
    if [ "$j" -lt "6" ]
    then python -m lib.bruteforce --output=./outputs/optimal/additive_bruteforce_kappa_0.1_m_${m[$j]}_seed_${seed[$i]} --m=${m[$j]} --kappa=0.1 --seed=${seed[$i]} --additive --gamma=$gamma
    fi
    python -m lib.thres --output=./outputs/optimal/additive_thres_kappa_0.1_m_${m[$j]}_seed_${seed[$i]} --m=${m[$j]} --kappa=0.1 --seed=${seed[$i]} --additive --gamma=$gamma
    for k in {0..2}
    do
      python -m lib.iterative --output=./outputs/optimal/additive_iterative_kappa_${kappa[$k]}_m_${m[$j]}_seed_${seed[$i]} --m=${m[$j]} --njobs=1 --kappa=${kappa[$k]} --seed=${seed[$i]} --additive --gamma=$gamma
      python -m lib.dp --output=./outputs/optimal/additive_dp_kappa_${kappa[$k]}_m_${m[$j]}_seed_${seed[$i]} --m=${m[$j]} --kappa=${kappa[$k]} --seed=${seed[$i]} --gamma=$gamma
    done
  done
done