"""
Unified TFN Training Script

This script provides a unified CLI for training any TFN or baseline model on any supported dataset and task type.
It uses the model and dataset registries for dynamic argument parsing, validation, and instantiation.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from tfn.model.registry import (
    get_model_config,
    validate_model_task_compatibility,
    get_required_params,
    get_optional_params,
    get_physics_params,
    get_model_defaults,
    validate_kernel_evolution,
)
from tfn.datasets.registry import DATASET_REGISTRY, get_dataset_config, get_dataset_default_params, validate_dataset_task_compatibility
import ast

# Import all dataset loader functions
from tfn.tfn_datasets.glue_loader import (
    load_sst2, load_mrpc, load_qqp, load_qnli, load_rte, load_cola, load_wnli, load_stsb
)
from tfn.tfn_datasets.climate_loader import (
    load_electricity_transformer, load_jena_climate, load_jena_climate_multi
)
from tfn.tfn_datasets.pg19_loader import create_pg19_dataloader
from tfn.tfn_datasets.arxiv_loader import load_arxiv
from tfn.tfn_datasets.ner_loader import load_conll2003
# Note: load_long_text and load_cifar10 may not exist yet - will handle in dry run

# Type map for common parameters
PARAM_TYPE_MAP = {
    'vocab_size': int,
    'embed_dim': int,
    'num_classes': int,
    'output_dim': int,
    'input_dim': int,
    'seq_len': int,
    'grid_size': int,
    'time_steps': int,
    'num_layers': int,
    'num_heads': int,
    'pos_dim': int,
    'val_split': int,
    'patch_size': int,
    'grid_height': int,
    'grid_width': int,
    'output_len': int,
    'num_filters': int,
    'dropout': float,
    'constraint_weight': float,
    'learning_rate': float,
    'lr': float,
    'bidirectional': lambda x: x.lower() in ('true', '1', 'yes'),
    'use_global_pooling': lambda x: x.lower() in ('true', '1', 'yes'),
    'use_physics_constraints': lambda x: x.lower() in ('true', '1', 'yes'),
    'filter_sizes': lambda x: [int(i) for i in x.split(',')],
}

# -----------------------------
# Dynamic CLI Argument Parsing
# -----------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Unified TFN Training Script")
    parser.add_argument('--task', type=str, required=True, 
                       choices=['classification', 'regression', 'time_series', 'language_modeling', 'vision', 'ner'], 
                       help='Task type')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--dry_run', action='store_true', help='If set, only test imports, argument parsing, and instantiation, then exit.')
    # Add universal params here; model/dataset-specific params will be added dynamically
    return parser

def infer_type(val, param=None):
    if param and param in PARAM_TYPE_MAP:
        return PARAM_TYPE_MAP[param]
    if isinstance(val, bool):
        return lambda x: x.lower() in ('true', '1', 'yes')
    if isinstance(val, int):
        return int
    if isinstance(val, float):
        return float
    if isinstance(val, list):
        return lambda x: [ast.literal_eval(i) for i in x.split(',')]
    return str

def add_dynamic_args(parser, model_name, dataset_name):
    model_config = get_model_config(model_name)
    for param in model_config.get('required_params', []):
        if not any(action.dest == param for action in parser._actions):
            default = model_config.get('defaults', {}).get(param, None)
            arg_type = infer_type(default, param) if default is not None else infer_type(None, param)
            parser.add_argument(f'--{param}', required=False, type=arg_type, default=default)
    for param in model_config.get('optional_params', []):
        if not any(action.dest == param for action in parser._actions):
            default = model_config.get('defaults', {}).get(param, None)
            arg_type = infer_type(default, param) if default is not None else infer_type(None, param)
            parser.add_argument(f'--{param}', required=False, type=arg_type, default=default)
    dataset_config = get_dataset_config(dataset_name)
    for param, value in dataset_config.get('default_params', {}).items():
        if not any(action.dest == param for action in parser._actions):
            arg_type = infer_type(value, param)
            parser.add_argument(f'--{param}', required=False, type=arg_type, default=value)
    return parser

# After args = parser.parse_args(), convert all model_kwargs and dataset_kwargs to correct type
# (should be handled by argparse, but add a final check)
def convert_types(kwargs):
    for k, v in kwargs.items():
        if k in PARAM_TYPE_MAP and not isinstance(v, PARAM_TYPE_MAP[k]):
            try:
                kwargs[k] = PARAM_TYPE_MAP[k](v)
            except Exception:
                pass
    return kwargs

# -----------------------------
# Main Training Logic
# -----------------------------
def main():
    # First parse the basic args to get model/dataset
    base_parser = build_parser()
    base_args, _ = base_parser.parse_known_args()
    # Add dynamic args
    parser = add_dynamic_args(base_parser, base_args.model, base_args.dataset)
    args = parser.parse_args()

    # If kernel/evolution args are present validate them
    if hasattr(args, 'kernel_type') and hasattr(args, 'evolution_type'):
        try:
            validate_kernel_evolution(args.kernel_type, args.evolution_type)
        except ValueError as e:
            sys.exit(f"[ConfigError] {e}")

    # Validate compatibility
    if not validate_model_task_compatibility(args.model, args.task):
        sys.exit(f"Model {args.model} is not compatible with task {args.task}.")
    if not validate_dataset_task_compatibility(args.dataset, args.task):
        sys.exit(f"Dataset {args.dataset} is not compatible with task {args.task}.")

    # Get model and dataset configs
    model_config = get_model_config(args.model)
    dataset_config = get_dataset_config(args.dataset)
    model_kwargs = {}
    # Fill in required and optional model params
    for param in model_config.get('required_params', []):
        val = getattr(args, param, None)
        if val is not None:
            model_kwargs[param] = val
        elif param in model_config.get('defaults', {}):
            model_kwargs[param] = model_config['defaults'][param]
        else:
            sys.exit(f"Missing required model parameter: {param}")
    for param in model_config.get('optional_params', []):
        val = getattr(args, param, None)
        if val is not None:
            model_kwargs[param] = val
        elif param in model_config.get('defaults', {}):
            model_kwargs[param] = model_config['defaults'][param]
    # Fill in dataset params
    dataset_kwargs = dataset_config.get('default_params', {}).copy()
    for k in dataset_kwargs:
        cli_val = getattr(args, k, None)
        if cli_val is not None:
            dataset_kwargs[k] = cli_val
    # Convert types for all parameters
    model_kwargs = convert_types(model_kwargs)
    dataset_kwargs = convert_types(dataset_kwargs)
    # Load dataset
    loader_fn_name = dataset_config['loader_function']
    loader_fn = globals().get(loader_fn_name)
    if loader_fn is None:
        sys.exit(f"Dataset loader function {loader_fn_name} not found.")
    # DRY RUN: Only instantiate, don't load full data
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Model: {args.model} ({model_config['class'].__name__})")
        print(f"Dataset: {args.dataset} (loader: {loader_fn_name})")
        print("Model parameters:")
        for k, v in model_kwargs.items():
            print(f"  {k}: {v} (type: {type(v)})")
            if v is None:
                print(f"    [WARNING] Parameter '{k}' is None!")
        print("Dataset parameters:")
        for k, v in dataset_kwargs.items():
            print(f"  {k}: {v} (type: {type(v)})")
            if v is None:
                print(f"    [WARNING] Parameter '{k}' is None!")
        # Try instantiating model and calling loader with dummy args
        try:
            model = model_config['class'](**model_kwargs)
            print("Model instantiation: SUCCESS")
        except Exception as e:
            print(f"Model instantiation: FAILED\n{e}")
        try:
            loader_result = loader_fn(**dataset_kwargs)
            print("Dataset loader: SUCCESS")
            if isinstance(loader_result, tuple):
                print(f"Loader returned {len(loader_result)} outputs.")
        except Exception as e:
            print(f"Dataset loader: FAILED\n{e}")
        print("=== END DRY RUN ===\n")
        sys.exit(0)

    # Load dataset
    train_data, val_data, *extra = loader_fn(**dataset_kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Add extra info (e.g., vocab size, num_classes) to model_kwargs if needed
    for k, v in zip(['vocab_size', 'num_classes', 'output_dim', 'num_tags'], extra):
        if k in model_kwargs:
            continue
        if v is not None:
            model_kwargs[k] = v
    # Instantiate model
    model_class = model_config['class']
    model = model_class(**model_kwargs).to(args.device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Loss function (task-specific)
    if args.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.task == 'regression' or args.task == 'time_series':
        criterion = torch.nn.MSELoss()
    elif args.task == 'language_modeling':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.task == 'ner':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        sys.exit(f"Unknown task type: {args.task}")
    # Training loop (simplified)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # Unpack batch (task-specific)
            if args.task == 'classification':
                input_ids, labels = batch
                logits = model(input_ids.to(args.device))
                loss = criterion(logits, labels.to(args.device))
            elif args.task == 'regression' or args.task == 'time_series':
                x, y = batch
                preds = model(x.to(args.device))
                loss = criterion(preds, y.to(args.device))
            elif args.task == 'language_modeling':
                input_ids, targets = batch
                logits = model(input_ids.to(args.device))
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(args.device))
            elif args.task == 'ner':
                input_ids, tags = batch
                logits = model(input_ids.to(args.device))
                loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(args.device))
            else:
                sys.exit(f"Unknown task type: {args.task}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {total_loss/len(train_loader):.4f}")
        # Validation (optional, simplified)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.task == 'classification':
                    input_ids, labels = batch
                    logits = model(input_ids.to(args.device))
                    loss = criterion(logits, labels.to(args.device))
                elif args.task == 'regression' or args.task == 'time_series':
                    x, y = batch
                    preds = model(x.to(args.device))
                    loss = criterion(preds, y.to(args.device))
                elif args.task == 'language_modeling':
                    input_ids, targets = batch
                    logits = model(input_ids.to(args.device))
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(args.device))
                elif args.task == 'ner':
                    input_ids, tags = batch
                    logits = model(input_ids.to(args.device))
                    loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(args.device))
                else:
                    continue
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    main() 