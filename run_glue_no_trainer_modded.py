# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from dotenv import load_dotenv
import os
import wandb
load_dotenv("./.env")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_steps_fraction",
        type=float,
        default=None, 
        help="Fraction of warmupsteps of total steps. Overrides `num_warmup_steps`"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--insert_dropout",
        type=float,
        default=-1,
        help="Change the dropout rate of the hidden layers during insertion.",
    )
    parser.add_argument(
        "--training_size",
        type=float,
        default=1,
        help="Change the fraction of training data used.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Change the beta1 value of the AdamW optimizer.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Change the beta2 value of the AdamW optimizer.",
    )

    parser.add_argument(
        "--catch_dropout",
        type=float,
        default=None,
        help="Change the dropout rate when catching a gradient. Not giving a value results in using the dropout value from the pretraining.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Set the number early stopping patience."
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0,
        help="Set the minimum loss delta for the early stopping."
    )
    parser.add_argument(
        "--original_gradient_fraction",
        type=float,
        default=0.5,
        help="Determines the fraction of the gradient resulting from the forward pass with dropout to be inserted."
    )
    parser.add_argument(
        "--param_config_id",
        type=str,
        default=None,
        help="Distinct ID by which the constellation of hyperparameters can be identified"
    )
    parser.add_argument(
        "--run_generation",
        type=str,
        default=None,
        help="The distinct generation from which the run comes"
    )

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

class Catch_Hook():
    def __init__(self, module):
        self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module, grad_input, grad_output):
        self.caught_grad = grad_output

    def close(self):
        self.hook.remove()

class Insert_Hook():
    def __init__(self, module, insertion_enabled = False, new_grad_output=None, original_gradient_fraction=0, batch_size=32):
        self.new_grad_output = new_grad_output
        # use prepend=True so that this is definitely the first hook being applied
        self.hook = module.register_full_backward_pre_hook(self.hook_fn,prepend=True)
        self.insertion_enabled = insertion_enabled
        assert (0 <= original_gradient_fraction <= 1), "Gradient fraction should be between 0 and 1"
        self.original_gradient_fraction = original_gradient_fraction
        self.vector_norms = []
        self.rescaled_diffs = []
        self.avg_diff_per_class = []
        self.avg_diff_all_classes = []
        self.batch_size = batch_size

    def hook_fn(self, module, grad_output):
        if self.insertion_enabled:
            grad_diff = grad_output[0] - self.new_grad_output[0]
            grad_diff_rescaled = grad_diff *self.batch_size
            self.rescaled_diffs.append(grad_diff_rescaled)
            self.vector_norms.append(torch.linalg.vector_norm(grad_diff_rescaled,dim=1))
            self.avg_diff_per_class.append(grad_diff_rescaled.abs().mean(dim=0))
            self.avg_diff_all_classes.append(self.avg_diff_per_class[-1].mean())
        # simply return the previously caught grad_output
        # this will replace the current grad_output (if prehook is used)
            if self.original_gradient_fraction > 0:
                ogf = self.original_gradient_fraction
                return ogf * grad_output + (1-ogf) * self.new_grad_output
            else:
                return self.new_grad_output

    def update_grad(self, new_grad_output):
        self.new_grad_output = new_grad_output

    def clear_lists(self):
        self.vector_norms = []
        self.rescaled_diffs = []
        self.avg_diff_per_class = []
        self.avg_diff_all_classes = []

    def close(self):
        self.hook.remove()

    def enable_insertion(self):
        self.insertion_enabled = True

    def disable_insertion(self):
        self.insertion_enabled = False

class early_stopping_callback:
  def __init__(self,metric_name,direction='min',min_delta=0,patience=5):
    '''
    `direction` is either 'min' or 'max' 
    '''
    self.metric_name=metric_name
    if direction not in ['max','min']:
        raise ValueError(
            f'Invalid direction: {direction}. Should be one of: max, min'
        )
    self.direction = direction
    self.sign = 1 if direction == 'min' else -1
    self.min_delta=min_delta
    self.patience=patience
    self.wait_steps=0
    self.improved_metric= False
    self.best_value=self.sign * float('inf')
    self.best_epoch=0
    self.best_step=0

  def check_early_stopping(self,value,epoch,step):
    delta = self.sign * (self.best_value - value)
    if delta >= self.min_delta: # improved metric
        self.best_value = value
        self.best_epoch = epoch
        self.best_step = step
        self.wait_steps = 0
        self.improved_metric = True        
    else: # not improved metric
        self.wait_steps += 1
        self.improved_metric = False
        if self.wait_steps >= self.patience:
            return True
    return False
  
def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    softmax = torch.nn.Softmax(dim=1)

    #++++++++++++++++++ attach hooks to the last layer ++++++++++++++++++++++++#
    layer = model.classifier
    hooks = {
        'catch_hook' : Catch_Hook(layer),
        'insert_hook' : Insert_Hook(layer,args.original_gradient_fraction, batch_size=args.per_device_train_batch_size)
    }

    # determine dropout_layers which will be activated later (skip input dropout)
    dropout_modules = [module for module in model.modules() if isinstance(module,torch.nn.Dropout)][1:]
    
    use_modded = not (args.catch_dropout is None)
    
    # set the dropout values for the catch run
    if use_modded:
        if 0 <= args.catch_dropout <= 1:
            catch_dropouts = [args.catch_dropout for _ in dropout_modules]
        else:
            catch_dropouts = [module.p for module in dropout_modules]
        

    # set the dropout values for the insertion run
    if  0 <= args.insert_dropout <= 1:
        insert_dropouts = [args.insert_dropout for _ in dropout_modules]
    else: 
        insert_dropouts = [module.p for module in dropout_modules]

    # initialize the dropout to the desired value. If no insertion is done, this dropout values
    # will be used as the standard dropout value.
    for module, p in zip(dropout_modules,insert_dropouts):
        module.p = p

    #++++++++++++++++++ \attach hooks to the last layer ++++++++++++++++++++++++#

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    #++++++++++++++++++ create early stopping callback  ++++++++++++++++++++++++#
    # Eval loss
    early_stoppings = [early_stopping_callback(
        metric_name='eval_loss',
        direction='min',
        min_delta=args.early_stopping_min_delta,
        patience=args.early_stopping_patience
    )]

    metric.add_batch(predictions=[1,1,0,0],references=[1,0,1,0])
    metric_names = list(metric.compute().keys())
    for metric_name in metric_names:
        early_stoppings.append(early_stopping_callback(
            metric_name=metric_name,
            direction='max',
            min_delta=args.early_stopping_min_delta,
            patience=args.early_stopping_patience
    ))
    #++++++++++++++++++ \create early stopping callback ++++++++++++++++++++++++#


    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    ########### Modify datasets #############
    if args.training_size != 1.0:
        pre_train_dataset = processed_datasets["train"]
        logger.info(f'Original number of training samples is: {len(pre_train_dataset)} with {sum(pre_train_dataset["labels"])/len(pre_train_dataset["labels"])} of the data in class 1')

        pre_eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
        logger.info(f'Original number of validation samples is: {len(pre_eval_dataset)} with {sum(pre_eval_dataset["labels"])/len(pre_eval_dataset["labels"])} of the data in class 1')
        
        target_train_size = int(args.training_size) if args.training_size > 1 else args.training_size
        train_indices, validation_indices, _, _ = train_test_split(
            range(len(pre_train_dataset)), # dummy indices
            pre_train_dataset["labels"], #
            stratify = pre_train_dataset["labels"],
            train_size = target_train_size,
            random_state = args.seed
        )

        # if training sizes are really small compared to the total number of samples (e.g. only 32 samples or 0.01 fraction),
        # there would be relatively many validation samples being added. This could lead to one huge evaluation loop for
        # an update only consisting 32 samples. This would incur a dramatic increase in evaluation and therefore training time,
        # even though the benefits of having more validation samples is relatively limited. Therefore, cap the number of
        # validation samples to be 4 times the original number of validation samples. Simply drop the rest.
        maximum_number_validation_samples = 4 * len(pre_eval_dataset)
        if len(validation_indices) > maximum_number_validation_samples:
            # create another stratified subset of the validation indices
            reduced_validation_indices, _, _, _ = train_test_split(
                validation_indices, # take subset of previous validation indices
                [pre_train_dataset["labels"][index] for index in validation_indices], #
                stratify = [pre_train_dataset["labels"][index] for index in validation_indices],
                train_size = maximum_number_validation_samples,
                random_state = args.seed
            )
        else:
            reduced_validation_indices = validation_indices

        train_dataset = Subset(pre_train_dataset,train_indices)
        logger.info(f'Removed {len(validation_indices)} samples from the training data. Remaining samples: {len(train_dataset)}.')

        eval_dataset = pre_eval_dataset
        # eval_dataset = ConcatDataset([pre_eval_dataset,Subset(pre_train_dataset,reduced_validation_indices)])
        # logger.info(f'Adding {len(reduced_validation_indices)} samples to from the training data to the validation data. New number of validation samples: {len(eval_dataset)}')
        
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        if len(train_indices) < total_batch_size:
            logger.info(f'Not enough training samples to fill a whole batch. Currently {len(train_dataset)} but require {args.per_device_train_batch_size}')
            missing_factor = int(total_batch_size/len(train_indices))
            train_dataset = ConcatDataset([train_dataset]*missing_factor)

            logger.info(f'Repeated training data {missing_factor} times. Training data now has {len(train_dataset)} samples.')
    else:
        train_dataset = processed_datasets["train"]
        logger.info(f'Number of training samples is: {len(train_dataset)} with {sum(train_dataset["labels"])/len(train_dataset["labels"])} of the data in class 1')
   
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
        logger.info(f'Number of validation samples is: {len(eval_dataset)} with {sum(eval_dataset["labels"])/len(eval_dataset["labels"])} of the data in class 1')
    ########### \Modify datasets #############


    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    adjusted_learning_rate = args.learning_rate * accelerator.num_processes
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=adjusted_learning_rate, betas=(args.beta1,args.beta2))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.warmup_steps_fraction is not None:
        if not (0 <= args.warmup_steps_fraction <= 1):
            raise ValueError(f'`warmup_steps_fraction` has to be a float in the interval [0,1]') 
        args.num_warmup_steps = int(args.warmup_steps_fraction * args.max_train_steps)
        overrode_num_warmup_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    modules_require_grad = {
        named_param[0] : named_param[1].requires_grad for named_param in model.named_parameters()
    }

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    if args.with_tracking:
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        run_name = "/".join(args.output_dir.split("/")[-2:])
        accelerator.init_trackers(project_name="fantastic-umbrella",
                                  config=experiment_config,
                                  init_kwargs={"wandb": {"entity": "Ricu",
                                                         "name": run_name},
                                               })
        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb").tracker
            wandb_tracker.watch(model)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # allow training to be stopped via variable
    # training_running = True

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        avg_train_p_max = 0
        avg_train_p_var = 0

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            model.train()
            if use_modded:
                # Stop syncing the gradients when freezing parts of the model.
                with accelerator.no_sync(model):
                    # #++++++++ catch gradient ++++++++#
                    # disable gradient insertion for catch run
                    hooks['insert_hook'].disable_insertion()

                    # set the dropout to the desired value during the catch run
                    for module, p in zip(dropout_modules,catch_dropouts):
                        module.p = p
                    
                    # freeze all layers except for the last
                    for named_param in model.named_parameters():
                        if 'classifier' not in named_param[0]:
                            named_param[1].requires_grad = False
                    

                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    # update gradient to be inserted
                    hooks['insert_hook'].update_grad(hooks['catch_hook'].caught_grad)

                    # set the dropout to the desired value during the insertion run
                    for module, p in zip(dropout_modules,insert_dropouts):
                        module.p = p
                    
                    # unfreeze all layers as they have been before
                    for named_param in model.named_parameters():
                        named_param[1].requires_grad = modules_require_grad[named_param[0]]

                    # re-enable gradient insertion for insertion run
                    hooks['insert_hook'].enable_insertion()
                    # #++++++++ \catch gradient ++++++++#

                
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            avg_train_p_max += softmax(outputs.logits.detach()).max(dim=1)[0].mean()
            avg_train_p_var += softmax(outputs.logits.detach()).var(dim=1).mean()

            # print(f'DEVICE {torch.cuda.current_device()}: cls: {list(layer.parameters())}')

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1         

            if args.with_tracking:
                accelerator.log({"learning_rate" : lr_scheduler.get_last_lr()[-1]})

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break


        model.eval()
        samples_seen = 0
        eval_loss = 0
        avg_eval_p_max = 0
        avg_eval_p_var = 0

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.detach().float()
                avg_eval_p_max += softmax(outputs.logits.detach()).max(dim=1)[0].mean()
                avg_eval_p_var += softmax(outputs.logits.detach()).var(dim=1).mean()

            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        
        eval_metrics = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metrics}")
        vector_norms = accelerator.gather_for_metrics(hooks["insert_hook"].vector_norms)
        if isinstance(vector_norms, list):
            vector_norms = torch.cat(vector_norms)
        avg_grad_diffs_per_class = accelerator.gather_for_metrics(hooks["insert_hook"].avg_diff_per_class)
        if isinstance(avg_grad_diffs_per_class, list):
            avg_grad_diffs_per_class = torch.stack(avg_grad_diffs_per_class).mean(dim=0)
        avg_grad_diffs_all_classes = accelerator.gather_for_metrics(hooks["insert_hook"].avg_diff_all_classes)
        if isinstance(avg_grad_diffs_all_classes, list):
            avg_grad_diffs_all_classes = torch.stack(avg_grad_diffs_all_classes).mean()
        hooks["insert_hook"].clear_lists()

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_loss": eval_loss.item() / len(eval_dataloader),
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "avg_train_p_max" : avg_train_p_max / len(train_dataloader),
                    "avg_eval_p_max" : avg_eval_p_max / len(eval_dataloader),
                    "avg_train_p_var" : avg_train_p_var / len(train_dataloader),
                    "avg_eval_p_var" : avg_eval_p_var / len(eval_dataloader),
                    "avg_grad_diff_all_classes" : avg_grad_diffs_all_classes,
                    "avg_grad_diff_per_class" : avg_grad_diffs_per_class,
                    "vector_norms" : vector_norms
                } | eval_metrics,
                step=completed_steps,
            )
            


        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        ### Early stoppings
        eval_metrics['eval_loss'] = torch.mean(accelerator.gather_for_metrics(eval_loss)).item() / len(eval_dataloader)
        metrics_ready_to_stop = [es.check_early_stopping(eval_metrics[es.metric_name],epoch,completed_steps) for es in early_stoppings]

        if all(metrics_ready_to_stop):
            accelerator.set_trigger()
            logger.info(f"Stopping early after epoch {epoch}.")
            for es in early_stoppings:
                logger.info(
                    'Achieved best value ({}) for metric {} at the end of epoch {}'.format(
                        es.best_value,
                        es.metric_name,
                        es.best_epoch
                    )
                )
        # update best values
        if args.with_tracking:
            for es in early_stoppings:
                if es.improved_metric and accelerator.is_main_process:
                    wandb_tracker.summary[f"best_{es.metric_name}"] = es.best_value
                    wandb_tracker.summary[f"best_{es.metric_name}_step"] = es.best_step
                    wandb_tracker.summary[f"best_{es.metric_name}_epoch"] = es.best_epoch

        ### Stop model if neccessary
        if accelerator.check_trigger():
            break
        
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(
        #     args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        # )
        if accelerator.is_main_process:
            # tokenizer.save_pretrained(args.output_dir) 
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        summaries = {f'best_{es.metric_name}' : es.best_value for es in early_stoppings}
        with open(os.path.join(args.output_dir, "run_overview.json"), "w") as f:   
            argument_dict = experiment_config
            argument_dict.update(summaries)
            json.dump(argument_dict,f)

if __name__ == "__main__":
    main()
