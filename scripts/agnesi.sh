m=200
sparsity=(200 198 196 194 192 190 175 150 125 100 75 50 25 0)
seed=(1 2 3 4 5 6 7 8 9 10)
gamma=0.3
max_iter=20
njobs=40
for i in {0..9}
do
  for j in {0..13}
  do
  # python -m lib.greedy_heur --output=./outputs/agnesi_heur_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=$m --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=1
  python -m lib.thres --output=./outputs/agnesi_thres_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=$m --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma
  python -m lib.iterative --output=./outputs/agnesi_iterative_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=$m --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=1
  python -m lib.iterative --output=./outputs/agnesi_iterative_parallel_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=$m --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=$njobs --max_iter=$max_iter
  done
done
