#!/bin/bash

SAMPLE_RATE=22050

# fetch_clip(videoID, startTime, endTime)
fetch_clip() {
  echo "Fetching $1 ($2 to $3)"

  outname="validation_dataVideo_dogs/$1_$2"

  if [ -f "${outname}.mp4.gz" ]; then
    echo "Already have it."
    return
  elif [ -f "${outname}.mp4" ]; then 
    echo "Already have it. But decompressed"
    return
  fi

  youtube-dl https://youtube.com/watch?v=$1 \
      -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' \
      --output "${outname}.%(ext)s" --no-check-certificate

  if [ $? -eq 0 ]; then
    STARTTIME=$2
    ENDTIME=$3

    echo "Start time: $STARTTIME"
    echo "End time: $ENDTIME"
    DIFFERENCE=$(($ENDTIME-$STARTTIME))
    echo "Difference: $DIFFERENCE"

    yes | ffmpeg -loglevel quiet -i "${outname}.mp4" -ss "$STARTTIME" -t "$DIFFERENCE" "${outname}_out.mp4"
    mv "./${outname}_out.mp4" "./$outname.mp4"
    # gzip "./$outname.mp4"
  else
    echo "SLEEP"
    sleep 1
    echo "SLEPT"
  fi
}


# Check parameters
if [ "$#" -ne 1 ]; then
     >&2 echo "Illegal number of parameters"
    exit 1
fi

# Check file
if [ ! -f "${1}" ]; then
     >&2 echo "File ${1} not found"
    exit 1
fi

FILE="${1}"

echo ""
echo ${FILE}

echo "Start"

while read line; do
    VIDEO=`echo ${line} | cut -d"," -f1`
    START=`echo ${line} | cut -d"," -f2`
    END=`echo ${line} | cut -d"," -f3`
    ENDNEW="${END/$'\r'/}"

    fetch_clip ${VIDEO} ${START} ${ENDNEW}

done < ${FILE}

echo "FINISHED"
