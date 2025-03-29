FKC_55=generated_samples/
FKC_75=generated_samples_beta_75/
FKC_55_01_09=generated_samples_beta_55_01_09/
SDXL=generated_images_sdxl_cfg7.5

for i in $FKC_55 $FKC_75 $FKC_55_01_09 $SDXL
do
  #python evaluation/evaluate_images.py $SCRATCH/$i/ --outfile $SCRATCH/geneval_results/${i}/results_val.jsonl --model-path $SCRATCH/geneval/ --valset-only
  echo $i
  python evaluation/summary_scores.py $SCRATCH/geneval_results/$i/results_val.jsonl
done
