#!/bin/bash
> $1.txt
name="python ../train.py"
name="3079"
if [ $# -eq 2 ]; then
    name=$2
    echo $2
fi

echo $name
while true;
    do ps aux | grep -w $name | grep -v grep | grep -v monitor.sh | awk '{print $5 " "$6}' >> $1.txt; echo ''  >> $1.txt;
    sleep 5;
done