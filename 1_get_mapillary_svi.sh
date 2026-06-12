TOTAL=236107
CHUNK=5000

START=$((1 * CHUNK))
END=$((START + CHUNK))
if [ "$END" -gt "$TOTAL" ]; then
  END="$TOTAL"
fi

echo "Task ${TASK_ID}: processing ${START}..${END}"

python ./scripts/svi.py --start "$START" --end "$END"
