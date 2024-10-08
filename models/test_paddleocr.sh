cd PaddleOCR/ppstructure
# 版面分析
python3 predict_system.py --layout_model_dir=inference/picodet_lcnet_x1_0_layout_infer                           --image_dir=./docs/table/1.png                           --output=../output                           --table=false                           --ocr=false --vis_font_path=../doc/fonts/simfang.ttf

# 表格识别
python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv3_det_infer                           --rec_model_dir=inference/ch_PP-OCRv3_rec_infer                           --table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer                           --image_dir=./docs/table/table.jpg                           --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt                           --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt                           --output=../output                           --vis_font_path=../doc/fonts/simfang.ttf                           --layout=false

#关键信息抽取
cd inference
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
cd ..
python3 predict_system.py \
  --kie_algorithm=LayoutXLM \
  --ser_model_dir=./inference/ser_vi_layoutxlm_xfund_infer \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../ppocr/utils/dict/kie_dict/xfund_class_list.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx" \
  --mode=kie

