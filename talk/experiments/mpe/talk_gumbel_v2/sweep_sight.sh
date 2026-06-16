for sight_range in 0.25 0.5 1.5 -1.0; do
  echo "=== sight_range=${sight_range} ==="
  python mappo_talk.py \
    "sight_range=${sight_range}" \
    
done
echo "Sweep complete."