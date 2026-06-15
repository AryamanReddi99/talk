for sight_range in 0.0 0.1 0.25 0.5 0.75 1.0 1.5 2.0 3.0 4.0 -1.0; do
  echo "=== sight_range=${sight_range} ==="
  python mappo.py \
    "sight_range=${sight_range}" \
    
done
echo "Sweep complete."