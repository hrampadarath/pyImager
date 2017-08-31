#!/usr/bin/env bash

NOW=$(date +"%d-%b-%y:%T")

#name of the imager to execute
Imager="pyImagerAproj" 

echo "executing  $Imager"
dir="./logs/"
LOGFILE="$dir$Imager-$NOW.log"
echo "Saving logs to $LOGFILE"

python $Imager.py -v -f makeImage.inputs #> $LOGFILE

echo "$Imager completed"



