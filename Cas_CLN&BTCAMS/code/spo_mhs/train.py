# _*_ coding:utf-8 _*_
import codecs
import json
import logging
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from code.spo_mhs import mhs_pointer as rel_bert
from code.spo_mhs.select_pointer_decoder import selection_decode
from code.utils.bert_optimization import BertAdam

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, spo_conf, tokenizer):

        self.args = args
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        self.model = rel_bert.MHSNet(args)

        self.model.to(self.device)
        if args.train_mode != "train":
            self.resume(args)

        if self.n_gpu > 1:
            logging.info('total gpu num is {}'.format(self.n_gpu))
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])

        if args.train_mode != "test":
            train_dataloader, dev_dataloader = data_loaders
            train_eval, dev_eval = examples
            self.eval_file_choice = {
                "train": train_eval,
                "dev": dev_eval,
            }
            self.data_loader_choice = {
                "train": train_dataloader,
                "dev": dev_dataloader,
            }
            self.optimizer = self.set_optimizer(args, self.model,
                                                train_steps=(int(
                                                    len(train_eval) / args.train_batch_size) + 1) * args.epoch_num)

        else:
            test_dataloader = data_loaders
            test_eval = examples
            self.eval_file_choice = {
                "test": test_eval
            }
            self.data_loader_choice = {
                "test": test_dataloader
            }

    def set_optimizer(self, args, model, train_steps=None):
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        logging.info('原始设置学习率设置')

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=train_steps)
        return optimizer

    def train(self, args):

        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        step_gap = 20
        for epoch in range(int(args.epoch_num)):

            global_loss, global_crf_loss, global_selection_loss = 0.0, 0.0, 0.0

            for step, batch in tqdm(enumerate(self.data_loader_choice[u"train"]), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):

                loss, crf_loss, selection_loss = self.forward(batch)

                if step % step_gap == 0:
                    global_loss += loss
                    global_crf_loss += crf_loss
                    global_selection_loss += selection_loss
                    current_loss = global_loss / step_gap
                    current_crf_loss = global_crf_loss / step_gap
                    current_selection_loss = global_selection_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}\tner:{}\trel:{}".format(step, len(
                            self.data_loader_choice["train"]),
                                                                                           epoch,
                                                                                           round(current_loss * 100, 5),
                                                                                           round(current_crf_loss * 100,
                                                                                                 5),
                                                                                           round(
                                                                                               current_selection_loss * 100,
                                                                                               5)))
                    global_loss, global_crf_loss, global_selection_loss = 0.0, 0.0, 0.0

            res_dev = self.eval_data_set("dev")
            if res_dev['f1'] >= best_f1:
                best_f1 = res_dev['f1']
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = self.model.module if hasattr(self.model,
                                                             'module') else self.model  # Only save the model it-self
                output_model_file = args.output + "/pytorch_model.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))
                patience_stop = 0
            else:
                patience_stop += 1
            if patience_stop >= args.patience_stop:
                return

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, batch, chosen=u'train', eval=False, answer_dict=None):

        batch = tuple(t.to(self.device) for t in batch)
        if not eval:
            passage_ids, segment_ids, ent_ids, rel_ids = batch
            loss, crf_loss, selection_loss = self.model(passage_ids=passage_ids, segment_ids=segment_ids,
                                                        ent_ids=ent_ids, rel_ids=rel_ids)
            if self.n_gpu > 1:
                loss = loss.mean()
                crf_loss = crf_loss.mean()
                selection_loss = selection_loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            crf_loss = crf_loss.item()
            selection_loss = selection_loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss, crf_loss, selection_loss
        else:
            p_ids, passage_ids, segment_ids = batch
            eval_file = self.eval_file_choice[chosen]
            ent_logits, rel_logits = self.model(passage_ids=passage_ids, segment_ids=segment_ids, is_eval=eval)
            answer_dict = self.convert_select_contour(eval_file, p_ids, passage_ids, ent_logits, rel_logits)
            return answer_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                answer_dict_ = self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
                answer_dict.update(answer_dict_)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))
        res = self.evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return res

    def predict_data_set(self, chosen="test"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                answer_dict_ = self.forward(batch, chosen, eval=True)
                answer_dict.update(answer_dict_)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        with codecs.open('user_data/res_data/'+self.args.res_path[:-5]+"_mhs.json", 'w', 'utf-8') as f:
            for key, value in answer_dict.items():
                entity_pred, spo_tuple_lst = value
                out_put = {}
                out_put['text'] = eval_file[key].context
                spo_lst = []
                for (s, p, o) in spo_tuple_lst:
                    spo_lst.append({"predicate": p, "subject": s, "object": {"@value": o}})
                out_put['spo_list'] = spo_lst

                json_str = json.dumps(out_put, ensure_ascii=False)
                f.write(json_str)
                f.write('\n')

    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):

        entity_em = 0
        entity_pred_num = 0
        entity_gold_num = 0
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for key, value in answer_dict.items():
            triple_gold = eval_file[key].gold_rel
            entity_gold = eval_file[key].gold_ent
            entity_pred, triple_pred = value

            entity_em += len(set(entity_pred) & set(entity_gold))
            entity_pred_num += len(set(entity_pred))
            entity_gold_num += len(set(entity_gold))

            R = set([spo for spo in triple_pred])
            T = set([spo for spo in triple_gold])
            X += len(R & T)
            Y += len(R)
            Z += len(T)

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        entity_precision = 100.0 * entity_em / entity_pred_num if entity_pred_num > 0 else 0.
        entity_recall = 100.0 * entity_em / entity_gold_num if entity_gold_num > 0 else 0.
        entity_f1 = 2 * entity_recall * entity_precision / (entity_recall + entity_precision) if (
                                                                                                         entity_recall + entity_precision) != 0 else 0.0

        print('============================================')
        print("{}/entity_em: {},\tentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, entity_em, entity_pred_num,
                                                                                   entity_gold_num))
        print(
            "{}/entity_f1: {}, \tentity_precision: {},\tentity_recall: {} ".format(chosen, entity_f1, entity_precision,
                                                                                   entity_recall))
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, X, Y, Z))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1 * 100, precision * 100,
                                                                recall * 100))
        return {'f1': f1, "recall": recall, "precision": precision}

    def convert_select_contour(self, eval_file, q_ids, input_ids, ent_logit, rel_logit):
        rel_pre = rel_logit > 0
        ent_pre = ent_logit
        answer_dict = selection_decode(q_ids, eval_file, ent_pre, rel_pre)
        return answer_dict
