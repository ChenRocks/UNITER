OUT_DIR=$1 
python inf_re.py \
    --txt_db /txt/refcoco_val.db:/txt/refcoco_testA.db:/txt/refcoco_testB.db \
    --img_db /img/re_coco_gt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --tmp_file re_exp/tmp_refcoco.txt

python inf_re.py \
        --txt_db /txt/refcoco_val.db:/txt/refcoco_testA.db:/txt/refcoco_testB.db \
        --img_db /img/re_coco_det \
        --output_dir $OUT_DIR  \
        --checkpoint best \
        --tmp_file re_exp/tmp_refcoco.txt
