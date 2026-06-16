DUCOR_DEFAULTS = {
    "epochs": 20,
    "warmup_steps": 0,
    "alpha": 0.75,
    "st2_open_alpha": 0.65,
    "st2_close_alpha": 0.80,
    "st2_ll_posterior_temperature": 2.0,
    "st2_ll_clean_rank_weight": 0.50,
    "st2_ll_clean_rank_temperature": 1.0,
    "st2_open_pseudo_weight_floor": 0.35,
    "st2_close_pseudo_weight_floor": 0.0,
    "st2_feature_temperature": 2.0,
    "st2_feature_margin_weight": 0.50,
    "st2_feature_margin_temperature": 0.07,
    "st2_missing_proto_weight": 0.20,
    "st2_selection_schedule": "0.2,0.4,0.6,1.0",
    "st2_open_selection_schedule": "0.5,0.7,0.9,1.0",
    "st2_close_selection_schedule": "0.1,0.3,0.6,1.0",
    "st2_warm_start_confidence": "auto",
    "st2_eval_pseudo_metrics": "no",
}

VQA_RAD_DUCOR_OVERRIDES = {
    "ducor_save_strategy": "last",
    "ducor_include_val_in_train": "no",
    "st2_open_selection_schedule": "0.2,0.3,0.4,0.5",
    "st2_close_selection_schedule": "0.1,0.2,0.3,0.4",
    "st2_open_pseudo_weight_floor": 0.0,
    "st2_missing_proto_weight": 0.1,
    "alpha": 0.85,
    "st2_open_alpha": 0.85,
    "st2_close_alpha": 0.90,
}


def add_ducor_arguments(parser):
    group = parser.add_argument_group("ducor")
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--lamda", type=float, default=0.5)
    group.add_argument("--alpha", type=float, default=0.5)
    group.add_argument("--seq_sim", type=str, default="mean", choices=("query", "mean"))
    group.add_argument("--st2_open_alpha", type=float, default=None)
    group.add_argument("--st2_close_alpha", type=float, default=None)
    group.add_argument("--st2_ll_posterior_temperature", type=float, default=1.0)
    group.add_argument("--st2_ll_clean_rank_weight", type=float, default=0.0)
    group.add_argument("--st2_ll_clean_rank_temperature", type=float, default=1.0)
    group.add_argument("--st2_open_pseudo_weight_floor", type=float, default=0.0)
    group.add_argument("--st2_close_pseudo_weight_floor", type=float, default=0.0)
    group.add_argument("--st2_feature_temperature", type=float, default=1.0)
    group.add_argument("--st2_feature_margin_weight", type=float, default=0.0)
    group.add_argument("--st2_feature_margin_temperature", type=float, default=0.1)
    group.add_argument("--st2_missing_proto_weight", type=float, default=1.0)
    group.add_argument("--st2_selection_schedule", type=str, default="1.0")
    group.add_argument("--st2_open_selection_schedule", type=str, default=None)
    group.add_argument("--st2_close_selection_schedule", type=str, default=None)
    group.add_argument("--st2_warm_start_confidence", type=str, default="no", choices=("no", "yes", "auto"))
    group.add_argument("--st2_eval_pseudo_metrics", type=str, default="no", choices=("no", "yes"))
    group.add_argument("--ducor_include_val_in_train", type=str, default="no", choices=("no", "yes"))
    group.add_argument("--ducor_save_strategy", type=str, default="val_acc", choices=("val_acc", "val_loss", "last"))


def apply_method_config(args):
    if args.method == "ducor":
        for key, value in DUCOR_DEFAULTS.items():
            setattr(args, key, value)
    return args


def apply_dataset_config(args):
    if args.method == "ducor" and args.dataset == "VQA_RAD":
        for key, value in VQA_RAD_DUCOR_OVERRIDES.items():
            setattr(args, key, value)
    return args
