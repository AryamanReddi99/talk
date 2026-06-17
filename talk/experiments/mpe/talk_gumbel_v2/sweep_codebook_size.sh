for codebook_size in 2 4 8 16 32; do
  echo "=== codebook_size=${codebook_size} ==="
  python mappo_talk_v2.py \
    "codebook_size=${codebook_size}" \
    "aux_coef=0.0" \
    "talk_config=_${codebook_size}_code_no_h_aux"
done
echo "Sweep complete."