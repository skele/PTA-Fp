#!/bin/bash
arg=$1
ofile=dump$1
line=$(($arg+1))
commandline=`sed -ne "${line}p" freqlist`
goodfit=1
amp=-13.0
#echo $amp
while [ "$goodfit" -ne 0 ]; do
    output=`/home/pbrem/PTA-Fp/pca_upper $ofile $commandline $amp J0030+0451 | grep DETECTED`
    goodfit=`echo $output | awk '{if ($2 < 0.93) print "1"; else if ($2 > 0.97) print "2"; else print "0";}'`
#    goodfit=`/home/pbrem/PTA-Fp/pca_upper $ofile $commandline $amp $thresh $3 $4 | grep DETECTED`
    echo $amp $goodfit $output
    if [ "$goodfit" -eq 1 ]
    then
	amp=$(awk -vamp="$amp" 'BEGIN {print amp+0.05}')
    fi
    if [ "$goodfit" -eq 2 ]
    then
	amp=$(awk -vamp="$amp" 'BEGIN {print amp-0.05}')
    fi
done
echo $amp $goodfit FINISHED