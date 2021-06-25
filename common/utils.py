"""
Date: 2021-06-01 22:29:43
LastEditors: GodK
LastEditTime: 2021-06-03 17:10:04
"""
import torch

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = ( y_pred - (1 - y_true) * 1e12 )  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()



class Preprocessor(object):
    def __init__(self, tokenizer):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
    
    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []
        
        inputs = self.tokenizer(text, add_special_tokens = False, return_offsets_mapping = True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens = False)

        for ent_span in entity_list:
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 寻找ent的token_span
            ent_token_len = len(ent2token)
            token_start_index = 0 if ent_token_len > 0 else - 1
            while token_start_index != -1:
                try:
                    token_start_index = text2tokens.index(ent2token[0], token_start_index)
                    if text2tokens[token_start_index:token_start_index + ent_token_len] == ent2token:
                        break
                    else:
                        token_start_index = text2tokens.index(ent2token[0], token_start_index + 1)
                except ValueError:
                    print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                    token_start_index = -1
            
            if token_start_index == -1:
                continue

            # 检查token_span与原span是否对应
            token_span = (token_start_index, token_start_index + ent_token_len - 1, ent_span[2])
            # XXX:对[UNK]的处理不完善
            assert '[UNK]' in ent2token or text[token2char_span_mapping[token_span[0]][0]:token2char_span_mapping[token_span[1]][1]]==ent, f'{ent}的token_span:{token_span}与原text:{text}不对应，请检查'
            ent2token_spans.append(token_span)
            
        
        return ent2token_spans

