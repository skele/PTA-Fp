#!/bin/bash
arg=$1
ofile=dump$1
line=$(($arg+1))
commandline=`sed -ne "${line}p" freqlist`
goodfit=1
amp=`sed -ne "${line}p" startamps`
#echo $amp
while [ "$goodfit" -ne 0 ]; do
    newfile=${ofile}_${amp}
    /home/pbrem/PTA-Fp/pca_upper $newfile $commandline $amp J0030+0451 J0034-0534 J0218+4232 J0610-2100 J0613-0200 J0621+1002 J0751+1807 J0900-3144 J1012+5307 J1022+1001 J1024-0719 J1455-3330 J1600-3053 J1640+2224 J1643-1224 J1713+0747 J1721-2457 J1730-2304 J1738+0333 J1744-1134 J1751-2857 J1801-1417 J1802-2124 J1804-2717 J1843-1113 J1853+1303 J1857+0943 J1909-3744 J1910+1256 J1911-1114 J1911+1347 J1918-0642 J1955+2908 J2010-1323 J2019+2425 J2033+1734 J2124-3358 J2145-0750 J2229+2643 J2317+1439 J2322+2057 &> /dev/null
    output=`../KStest/95percent.py $commandline $newfile`
    goodfit=`echo $output | awk '{if ($1 > 0.95) print "0"; else print "1";}'`
    echo $amp $goodfit $output
    if [ "$goodfit" -eq 1 ]
    then
	amp=$(awk -vamp="$amp" 'BEGIN {print amp+0.1}')
    fi
done
echo $amp $goodfit FINISHED
