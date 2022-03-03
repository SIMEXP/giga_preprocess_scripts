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
    for f in $files; do
        file_basename=$(basename $f);
        for a in schaefer mist segmented_difumo; do
            echo -e "$dataset\t$file_basename\t$a" >> all_jobs.tsv
        done;
    done;
done

# split to 40 batches; 400 processes max each batch
split -d -l 400 all_jobs.tsv batch_ 

echo "Created batch files"

for i in $(seq -f "%02g" 0 39); do
    while read -r line; do
        IFS=$'\t' read dataset file_basename atlas <<< $line
        echo $line
        python giga_preprocess/giga_timeseries.py -i $GIGA_DIR -d $dataset -s $file_basename -o $OUTPUT_ROOT -a $atlas 
    done < batch_$i > logs/batch_$i.log &
done
