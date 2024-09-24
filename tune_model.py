import os
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig
import bitsandbytes as bnb

from config import *
from utils import *


dpo_data_path = "./dpo_data.json"
new_model = "/mnt/d/work/models/google-1.1-dpo"


def chatml_format(example):

    # Format instruction
    message = [{"role": "user", "content": example['prompt']}, {"role": "assistant", "content": example['chosen']}]
    # prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    # Format chosen answer
    # chosen = example['chosen'] +  tokenizer.eos_token

    # # Format rejected answer
    # rejected = example['rejected'] +  tokenizer.eos_token

    # return {
    #     "prompt": prompt,
    #     "chosen": chosen,
    #     "rejected": rejected,
    # }
    return {"text": prompt}

# Load dataset
# dataset = load_dataset("Intel/orca_dpo_pairs")['train']
dataset = Dataset.from_dict(read_json(dpo_data_path))

# Save columns
original_columns = dataset.column_names

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)



# Print sample
print(dataset[1])
print()
print(dataset)
print()
# LoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_use_double_quant= False,
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map={"":0}, 
    low_cpu_mem_usage = True)


model.config.use_cache = False
print("Loaded first model")


def train_dpo_model():
    dpo_config = DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=200,
        save_strategy="no",
        logging_steps=1,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True,
        ref_model_init_kwargs=None,
        model_init_kwargs=None
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model = None,
        # args=training_args,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
        # force_use_ref_model=True
    )

    # Fine-tune model with DPO
    print("Starting to train")
    dpo_trainer.train()

    # dpo_trainer.model.save_pretrained("gemma-1.1-final_checkpoint")
    # tokenizer.save_pretrained("final_checkpoint")
    dpo_trainer.save_model(new_model)
    print("Finished training a DPO model")



def train_sft_model():

    # Training arguments
    # training_args = TrainingArguments(
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=2,
    #     gradient_checkpointing=True,
    #     learning_rate=5e-5,
    #     lr_scheduler_type="cosine",
    #     max_steps=200,
    #     dataset_text_field="text",
    #     save_strategy="no",
    #     logging_steps=1,
    #     output_dir=new_model,
    #     optim="paged_adamw_32bit",
    #     warmup_steps=100,
    #     bf16=True,
    # )   


    sft_config = SFTConfig(
        output_dir="./dummy_nopadeos_leftpad",
        # num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=800,
        dataset_text_field="text",
        save_strategy="steps",
        save_steps = 60,
        logging_steps=10,
        # optim="paged_adamw_32bit",
        optim="paged_adamw_8bit",
        warmup_steps=100,
        bf16=False,
        fp16=False
    )


    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        # dataset_kwargs={
        #     "add_special_tokens": False,  # We template with special tokens
        #     "append_concat_token": False, # No need to add additional separator token
        # }
    )

    print("Starting to train ...")
    sft_trainer.train()
    sft_trainer.save_model(new_model)

    print("Training Finished")

train_sft_model()
