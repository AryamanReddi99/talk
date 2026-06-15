for comm_range in 0.0 3.0 6.0 12.0 24.0; do
  echo "=== comm_range=${comm_range} ==="
  python mappo_tarmac.py \
    "comm_range=${comm_range}" \
    
done
echo "Sweep complete."