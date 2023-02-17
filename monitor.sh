#!/bin/bash
> $2.txt
while true;
    do ps aux | grep -w "python ../train.py" | grep -v grep | grep -v monitor.sh | awk '{print $5 " "$6}' >> $2.txt ;
    sleep 10;
done
