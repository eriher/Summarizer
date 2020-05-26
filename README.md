# Summarizer
Automatic Text Summarization using pre-trained language models. 

A Master Thesis Project in Computer Science by Erik Hermansson and Charlotte Boddien.

Paper: https://hdl.handle.net/20.500.12380/300719

This repo contains code for the project.

## Requirements
PyTorch

Cuda

TensorBoardx

Tqdm

transformers

numpy

scipy

sentence-transformers

rouge (this one,https://pypi.org/project/rouge/)

## Training/Fine-tuning
Fine-tuning is located in suminarizer_finetune.py. There are several input parameters explained there.

## Some examples:
This will run fine-tuning for roberta based model. This was ran on vera gpus with 32 gb of gpu memory.

> python summarizer_finetune.py --data_dir ../cnndm_data/cnndm --temp_dir ../temp --output_dir ../suminarizer_output/roberta --num_train_epochs 4 --per_gpu_train_batch_size 36 --num_workers 4 --fp16 True --lr 2e-5 --model_name roberta-base --beta2 0.98 --eval_save_steps 4000

This is a lighter example, it will run fine-tuning for distilbert based model, using per gpu batch size of 8 with gradient accumulation of 4 resulting in total batch size of 32. Could run this on 8gb gpu.

> python summarizer_finetune.py --data_dir ../cnndm_data/cnndm --temp_dir ../temp --output_dir ../suminarizer_output/distilbert --num_train_epochs 4 --per_gpu_train_batch_size 8 --gradient_accumulation 4 --num_workers 4 --lr 2e-5 --model_name distilbert-base-uncased --eval_save_steps 4000

## Validating checkpoints
This example will evaluate all checkpoints in folder and combine top three into single checkpoints and output results to output dir

> python summarizer_eval.py --model_path ../path/to/model --model_name name_of_transformers_model --data_dir D:/cnndm_data_ext/cnndm --output_dir ../path/for/output --validate True

## Evaluating on CNN/DM
This example will evaluate model on CNN/DM test portion and output results to output dir

> python summarizer_eval.py --model_path ../path/to/model --model_name name_of_transformers_model --data_dir ../path/to/data/cnndm  --output_dir ../path/for/output --test True

## The validation and evaluation can be combined
This example will evaluate all checkpoints in folder and perform testing on the combined model

> python summarizer_eval.py --model_path ../path/to/model --model_name name_of_transformers_model --data_dir D:/cnndm_data_ext/cnndm --output_dir ../path/for/output --validate True --test True


## Generating Summaries
Expects input data to be one or multiple txt files with content of text with one sentence per line. Outputs a txt file for each input txt file to cand_dir.

> python summarizer_eval.py --model_path ../path/to/model --model_name name_of_transformers_model --data_dir ../path/to/data --cand_dir ../path/to/save/summaries --gen True

## Evaluate Summaries
Provide two directories with matching txt files, one with reference summaries ref_dir and one with candidate summaries(generated summaries) cand_dir. This will run ROUGE, and some other metrics on the matching files in the folders. Results to output_folder

> python summarizer_eval.py --cand_folder ../results/bertsum/gen_no_tri --ref_folder ../text/ref1 --test_folders True --output_dir ../results/bertsum
