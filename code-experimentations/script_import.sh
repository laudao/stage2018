scp big:k*.tar ./
for x in k*.tar
do
    filename=$(basename $x)
    filename="${filename%.*}"
    betterfilename="${filename%?}"
    mkdir $betterfilename
    tar xvf $x -C $betterfilename
done

#scp big:k5examples1.tar k5examples1.tar
#scp big:k5examples2.tar k5examples2.tar
#mkdir examples_k5
#tar xvf k5examples1.tar -C examples_k5
#tar xvf k5examples2.tar -C examples_k5
#
#scp big:k5pairs1.tar k5pairs1.tar
#scp big:k5pairs2.tar k5pairs2.tar
#mkdir pairs_k5
#tar xvf k5pairs1.tar -C pairs_k5
#tar xvf k5pairs2.tar -C pairs_k5
#
#scp big:k5eval1.tar k5eval1.tar
#scp big:k5eval2.tar k5eval2.tar
#mkdir eval_k5
#tar xvf k5eval1.tar -C eval_k5
#tar xvf k5eval2.tar -C eval_k5
#
#scp big:k5depth1.tar k5depth1.tar
#scp big:k5depth2.tar k5depth2.tar
#mkdir depth_k5
#tar xvf k5depth1.tar -C depth_k5
#tar xvf k5depth2.tar -C depth_k5
#
#scp big:k5leaves1.tar k5leaves1.tar
#scp big:k5leaves2.tar k5leaves2.tar
#mkdir leaves_k5
#tar xvf k5leaves1.tar -C leaves_k5
#tar xvf k5leaves2.tar -C leaves_k5
#
#scp big:k5eval1.tar k5eval1.tar
#scp big:k5eval2.tar k5eval2.tar
#mkdir eval_k5
#tar xvf k5eval1.tar -C eval_k5
#tar xvf k5eval2.tar -C eval_k5
