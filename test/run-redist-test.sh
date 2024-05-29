#!/usr/bin/bash


mtx=$1
startnodes=$2
downnodes=$3
startppn=$4
downppn=$5

>debug-down.out
>$mtx-down.mtx

>debug-up.out
>$mtx-up.mtx

>$mtx-correct.mtx
mpirun -n $startppn ../build/test/tuning-redist --matpath redist_test_matrices/$mtx/$mtx.mtx --startnodes $startnodes --downnodes $downnodes --startppn $startppn --downppn $downppn


DIFFUP=$(diff ${mtx}-correct.mtx ${mtx}-up.mtx)
if [ "$DIFFUP" == "" ]; then 
    echo "Scaling up test passed!"
    rm  $mtx-up.mtx
else
    echo "Scaling up test failed"
fi

DIFFDOWN=$(diff ${mtx}-correct.mtx ${mtx}-down.mtx)
if [ "$DIFFDOWN" == "" ]; then
    echo "Scaling down test passed!"
    rm $mtx-down.mtx
else
    echo "Scaling down test failed"
fi


