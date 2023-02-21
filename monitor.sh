#!/bin/bash
filename=monitor/$1.txt
> $filename
name="../train.py"
if [ $# -eq 2 ]; then
    name=$2
fi

echo $name
while true;
    time=`date '+%Y%m%d%H%M%S'`;
    do echo $time  >> $filename; ps aux | grep -w $name | grep -v grep | grep -v monitor.sh | awk '{print $5 " "$6}' >> $filename; echo ''  >> $filename;
    sleep 5;
done