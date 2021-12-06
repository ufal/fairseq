# Preprocess/binarize the data
TEXT=corp/
python3 fairseq_cli/preprocess.py --cpu  --source-lang en1_shuf en2  --target-lang cs \
    --trainpref $TEXT/train_5M --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/encs_5M_1st_random    \
    --workers 4 
