#!/bin/bash

for C in 2 4 6 8 10 12 14 16 18 20; do
    zeros=`seq -s "+" $C | sed 's/[0-9]//g' | sed 's/+/0/g'`
    C="0.${zeros}1"
    echo "*** C = $C ***"
    ./svm-train -t 0 -q -v 5 -c $C data/corpus.svm /dev/null
done