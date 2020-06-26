m=(4 5 6 7 8 9 100 120 140 160 180 200)
sparsity=(1 1 2 2 2 2 25 30 35 40 45 50)
seed=(1 2 3 4 5 6 7 8)
gamma=0.3
njobs=4
max_iter=20
for i in {0..7}
do
  for j in {0..11}
  do
  if [ "$j" -lt "6" ]
  then python -m lib.bruteforce --output=./outputs/khayyam_bruteforce_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma
  fi
  python -m lib.greedy_heur --output=./outputs/khayyam_heur_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=$njobs
  python -m lib.thres --output=./outputs/khayyam_thres_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma
  python -m lib.iterative --output=./outputs/khayyam_iterative_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=1
  python -m lib.iterative --output=./outputs/khayyam_iterative_parallel_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=$njobs --max_iter=$max_iter
  done
done

