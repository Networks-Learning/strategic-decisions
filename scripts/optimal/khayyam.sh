m=(4 5 6 7 8 9 100 200 300 400 500 600)
sparsity=(1 1 2 2 2 2 25 50 75 100 125 150)
seed=(1 2 3 4 5 6 7 8 9 10)
gamma=0.3
for i in {0..9}
do
  for j in {0..11}
  do
  if [ "$j" -lt "6" ]
  then python -m lib.bruteforce --output=./outputs/optimal/khayyam_bruteforce_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma
  fi
  python -m lib.thres --output=./outputs/optimal/khayyam_thres_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma
  python -m lib.iterative --output=./outputs/optimal/khayyam_iterative_m_${m[$j]}_sparsity_${sparsity[$j]}_seed_${seed[$i]} --m=${m[$j]} --sparsity=${sparsity[$j]} --seed=${seed[$i]} --gamma=$gamma --njobs=1
  done
done

