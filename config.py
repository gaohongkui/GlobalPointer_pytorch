"""
Date: 2021-06-01 17:18:25
LastEditors: GodK
"""
import time

common = {
    "exp_name": "cluener",
    "encoder": "BERT",
    "data_home": "./datasets",
    "bert_path": "./pretrained_models/bert-base-chinese",  # bert-base-chinese or other plm from https://huggingface.co/models
    "run_type": "train",  # train, eval
    "f1_2_save": 0.5,  # 存模型的最低f1值
    "logger": "default"  # wandb or default，default意味着只输出日志到控制台
}

# wandb的配置，只有在logger=wandb时生效。用于可视化训练过程
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

train_config = {
    "train_data": "train.json",
    "valid_data": "dev.json",
    "test_data": "dev.json",
    "ent2id": "ent2id.json",
    "path_to_save_model": "./outputs",  # 在logger不是wandb时生效
    "hyper_parameters": {
        "lr": 2e-5,
        "batch_size": 64,
        "epochs": 50,
        "seed": 2333,
        "max_seq_len": 128,
        "scheduler": "CAWR"  # CAWR, Step, None
    }
}

eval_config = {
    "model_state_dir": "./outputs/cluener/",  # 预测时注意填写模型路径（时间tag文件夹）
    "run_id": "",
    "last_k_model": 1,  # 取倒数第几个model_state
    "predict_data": "test.json",
    "ent2id": "ent2id.json",
    "save_res_dir": "./results",
    "hyper_parameters": {
        "batch_size": 16,
        "max_seq_len": 512,
    }

}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 200,
}

# ---------------------------------------------
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}
