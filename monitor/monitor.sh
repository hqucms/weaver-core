#!/bin/bash
> $1.txt
name="python ../train.py"
if [ $# -eq 2 ]; then
    name=$2
fi

echo $name
while true;
    time=`date '+%Y%m%d%H%M%S'`;
    do echo $time  >> monitor/$1.txt; ps aux | grep -w $name | grep -v grep | grep -v monitor.sh | awk '{print $5 " "$6}' >> monitor/$1.txt; echo ''  >> monitor/$1.txt;
    sleep 5;
done