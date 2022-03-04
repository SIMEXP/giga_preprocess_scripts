GIGA_DIR='/data/cisl/giga_preprocessing/preprocessed_data'
OUTPUT_ROOT='/data/cisl/preprocessed_data/giga_timeseries'
datasets='adni ccna cimaq oasis'

cd ~/simexp/hwang/giga_preprocess_scripts
source env/bin/activate

rm batch_*
rm all_jobs.tsv

echo "get all processes to be checked"

for dataset in $datasets; do
    files=$(ls $GIGA_DIR/${dataset}_preprocess/resample/fmri*.nii.gz);
    for a in mist segmented_difumo schaefer; do
        for f in $files; do
            file_basename=$(basename $f);
            echo -e "$dataset\t$file_basename\t$a" >> all_jobs.tsv
        done;
    done;
done

# split to 40 batches; 400 processes max each batch
split -d -l 400 all_jobs.tsv batch_ 

echo "Created batch files"

for i in $(seq 0 39); do
    printf -v j "%02d" $i
    i=$((i+1))
    echo $i
    while read -r line; do
        IFS=$'\t' read dataset file_basename atlas <<< $line
        taskset --cpu-list $i python giga_preprocess/giga_timeseries.py -i $GIGA_DIR -d $dataset -s $file_basename -o $OUTPUT_ROOT -a $atlas 
    done < batch_$j > logs/batch_$j.log &
done
