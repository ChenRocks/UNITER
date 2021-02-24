OUT_DIR=$1 
python inf_re.py \
    --txt_db /txt/refcocog_val.db:/txt/refcocog_test.db \
    --img_db /img/re_coco_gt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --tmp_file re_exp/tmp_refcocog.txt

python inf_re.py \
        --txt_db /txt/refcocog_val.db:/txt/refcocog_test.db \
        --img_db /img/re_coco_det \
        --output_dir $OUT_DIR  \
        --checkpoint best \
        --tmp_file re_exp/tmp_refcocog.txt
