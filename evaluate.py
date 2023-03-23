"""
Date: 2021-06-11 13:54:00
LastEditors: GodK
LastEditTime: 2021-07-19 21:53:18
"""
import os
import config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
import numpy as np

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)


def load_data(data_path, data_type="predict"):
    if data_type == "predict":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                datas.append(line)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)


def data_generator(data_type="predict"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "predict":
        predict_data_path = os.path.join(config["data_home"], config["exp_name"], config["predict_data"])
        predict_data = load_data(predict_data_path, "predict")

    all_data = predict_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer)

    if data_type == "predict":
        predict_dataloader = DataLoader(MyDataset(predict_data),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id,
                                                                                    data_type="predict")
                                     )
        return predict_dataloader


def decode_ent(text, pred_matrix, tokenizer, threshold=0):
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)
    return ent_list


def predict(dataloader, model):
    predict_res = []

    model.eval()
    for batch_data in dataloader:
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                       batch_attention_mask.to(device),
                                                                       batch_token_type_ids.to(device),
                                                                       )
        with torch.no_grad():
            batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        for ind in range(len(batch_samples)):
            gold_sample = batch_samples[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            pred_matrix = batch_logits[ind]
            labels = decode_ent(text, pred_matrix, tokenizer)
            predict_res.append({"id": text_id, "text": text, "label": labels})
    return predict_res


def load_model():
    model_state_dir = config["model_state_dir"]
    model_state_list = sorted(filter(lambda x: "model_state" in x, os.listdir(model_state_dir)),
                              key=lambda x: int(x.split(".")[0].split("_")[-1]))
    last_k_model = config["last_k_model"]
    model_state_path = os.path.join(model_state_dir, model_state_list[-last_k_model])

    encoder = BertModel.from_pretrained(config["bert_path"])
    model = GlobalPointer(encoder, ent_type_size, 64)
    model.load_state_dict(torch.load(model_state_path))
    model = model.to(device)

    return model


def evaluate():
    predict_dataloader = data_generator(data_type="predict")

    model = load_model()

    predict_res = predict(predict_dataloader, model)

    if not os.path.exists(os.path.join(config["save_res_dir"], config["exp_name"])):
        os.mkdir(os.path.join(config["save_res_dir"], config["exp_name"]))
    save_path = os.path.join(config["save_res_dir"], config["exp_name"], "predict_result.json")
    # json.dump(predict_res, open(save_path, "w", encoding="utf-8"), ensure_ascii=False)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in predict_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    evaluate()
