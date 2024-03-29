GIGA_DIR='/data/cisl/giga_preprocessing/preprocessed_data'
OUTPUT_ROOT='/data/cisl/preprocessed_data/giga_timeseries'

datasets='adni ccna cimaq oasis'

cd ~/simexp/hwang/giga_preprocess_scripts
source env/bin/activate

echo "get all processes to be checked"

# for dataset in $datasets; do
#     files=$(ls $GIGA_DIR/${dataset}_preprocess/resample/fmri*.nii.gz);
#     for f in $files; do
#         file_basename=$(basename $f);
#         echo -e "$dataset\t$file_basename" >> all_files.tsv
#     done;
# done

echo "Created batch files"

split -d -l 200 all_files.tsv batch_files/difumo_batch_ 
for i in $(seq 0 26); do
    printf -v j "%02d" $i
    i=$((i+1))
    echo $i
    while read -r line; do
        IFS=$'\t' read dataset file_basename atlas <<< $line
        taskset --cpu-list $i python giga_preprocess/giga_timeseries.py -i $GIGA_DIR -d $dataset -s $file_basename -o $OUTPUT_ROOT >> logs/$dataset.$file_basename.log
    done < batch_files/difumo_batch_$j &
done
