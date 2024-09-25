# run a new pipeline
python run_pipeline.py --mode run  --save_load_path './cc_100k' --n_samples 100000 --build_hf_ds
# load existing pipeline
# python run_pipeline.py --mode load --save_load_path './cc_100k' --build_hf_ds
# # inference mode on new texts from an input dataset
# python run_pipeline.py --mode infer --save_load_path './cc_100k'  --n_samples <NB_INFERENCE_SAMPLES> --input_dataset <HF_DATA_FOR_INFERENCE>