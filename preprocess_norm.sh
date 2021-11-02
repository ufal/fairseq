# Preprocess/binarize the data
TEXT=corp/
python3 fairseq_cli/preprocess.py --cpu  --source-lang en1  --target-lang cs \
    --trainpref $TEXT/train_5M --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/encs_5M_norm    \
    --workers 4 
