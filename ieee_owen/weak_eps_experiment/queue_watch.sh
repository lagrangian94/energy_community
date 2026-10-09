#!/bin/bash
# Monitor feed for the main KL queue: one line per finished run, failure, stall stop,
# silent log (>25 min) or low memory; a newer instance takes over from older ones.
#   bash ieee_owen/weak_eps_experiment/queue_watch.sh
# Emits one line per event of the main KL queue; state persists across re-arms.
O=/c/Users/user/energy_community/ieee_owen/weak_eps_experiment/stochastic/main17_xi
ST=/c/Users/user/energy_community/ieee_owen/weak_eps_experiment/stochastic/main17_xi/.watch_state
mkdir -p $ST
[ -f $ST/rows ] || { [ -f $O/runs.csv ] && wc -l < $O/runs.csv || echo 1; } > $ST/rows
touch $ST/alerted
# a newer instance takes over: older ones see another id and exit
ME=$$-$RANDOM; echo $ME > $ST/owner
while true; do
  [ "$(cat $ST/owner)" = "$ME" ] || exit 0
  # finished runs
  n=$( [ -f $O/runs.csv ] && wc -l < $O/runs.csv || echo 1 ); seen=$(cat $ST/rows)
  if [ "$n" -gt "$seen" ]; then
    tail -n $((n - seen)) $O/runs.csv | while IFS=, read fin nn sc day r ex wall pp pw mf kd; do
      log=$O/logs/n${nn}_S${sc}_day${day}.log
      eps=$(grep -oE "eps_LR [0-9.e+-]+" $log 2>/dev/null | tail -1)
      st=$(grep -oE "status [a-z]+  rounds|rounds [0-9]+  status [a-z]+" $log 2>/dev/null | tail -1)
      stall=$(grep -c "without LB progress" $log 2>/dev/null)
      tag="DONE"; { [ "$ex" != "0" ] || [ "$kd" = "1" ]; } && tag="ALERT failed"
      echo "$tag n=$nn S=$sc day=$day wall=${wall}s exit=$ex killed=$kd peak=${pp}GB minfree=${mf}GB $eps ${st} stallstop=$stall"
    done
    echo $n > $ST/rows
  fi
  # current run
  cur=$(tr -d '\000' < $O/queue.log | grep -E "^[0-9-]+ [0-9:]+  n=" | tail -1)   # PowerShell >> appends UTF-16
  key=$(echo "$cur" | grep -oE "n=[0-9]+ \|Omega\|=[0-9]+ day [0-9]+" | tr -d '| ' )
  nn=$(echo "$cur" | grep -oE "n=[0-9]+" | cut -d= -f2); sc=$(echo "$cur" | grep -oE "Omega\|=[0-9]+" | cut -d= -f2); day=$(echo "$cur" | grep -oE "day [0-9]+" | cut -d' ' -f2)
  log=$O/logs/n${nn}_S${sc}_day${day}.log; elog=$O/logs/n${nn}_S${sc}_day${day}_ef_gurobi.log
  if [ -f "$log" ]; then
    if grep -qE "Traceback|Error|error:" $log && ! grep -q "tb:$key" $ST/alerted; then
      echo "ALERT traceback in $key: $(grep -E 'Error|error:' $log | tail -1 | cut -c1-200)"; echo "tb:$key" >> $ST/alerted
    fi
    if grep -q "without LB progress" $log && ! grep -q "ss:$key" $ST/alerted; then
      echo "NOTE stall stop in $key: $(grep 'without LB progress' $log | tail -1 | cut -c1-160)"; echo "ss:$key" >> $ST/alerted
    fi
    last=$(stat -c %Y $log); [ -f "$elog" ] && le=$(stat -c %Y $elog) && [ $le -gt $last ] && last=$le
    idle=$(( $(date +%s) - last ))
    if [ $idle -gt 1500 ] && ! grep -q "idle:$key:$((idle/1500))" $ST/alerted; then
      echo "ALERT no log output for $((idle/60)) min in $key"; echo "idle:$key:$((idle/1500))" >> $ST/alerted
    fi
  fi
  free=$(powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,2)" 2>/dev/null | tr -d '\r')
  if [ -n "$free" ] && awk "BEGIN{exit !($free < 1.5)}" && ! grep -q "mem:$key" $ST/alerted; then
    echo "ALERT available memory ${free} GB during $key"; echo "mem:$key" >> $ST/alerted
  fi
  if tr -d '\000\r' < $O/queue.log | tail -1 | grep -q "QUEUE DONE"; then echo "QUEUE DONE"; exit 0; fi
  sleep 60
done
