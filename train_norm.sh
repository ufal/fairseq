rm checkpoints/*
export PYTHONPATH=//home/large/data/models/marian/encz_exp/fairseq-multisrc
export CUDA_VISIBLE_DEVICES=1
python3 fairseq_cli/train.py \
    data-bin/encs_5M_norm \
    --arch transformer --share-decoder-input-output-embed --task translation \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
     --dropout 0.3 --weight-decay 0.0001 --max-epoch 20  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 512  --ddp-backend=no_c10d  \
    --eval-bleu --update-freq 4 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
