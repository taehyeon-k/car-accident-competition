#!/bin/bash
# Keep MAX experiments running; each line of runs/stage3_v2/queue.txt is "NAME --set k=v ...".
cd /workspace/car-accident
Q=runs/stage3_v2/queue.txt; MAX=${MAX:-2}
touch $Q
count_runs() {
  # Distinct run names (dataloader workers share the parent's command line).
  pgrep -fa "[s]tage3.experiments.run --name" | sed -E 's/.*--name ([^ ]+).*/\1/' | sort -u | wc -l
}
while true; do
  if [ "$(count_runs)" -lt "$MAX" ] && [ -s $Q ]; then
    line=$(head -1 $Q); sed -i 1d $Q
    [ -n "$line" ] && (eval stage3/experiments/exp.sh $line &) && echo "$(date +%T) start $line" >> runs/stage3_v2/logs/queue.log
    sleep 30
  fi
  sleep 15
done
