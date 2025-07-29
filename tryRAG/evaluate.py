"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507231725
"""

import json
import evaluate
from tqdm import tqdm
import random
import os

DATASET_PATH_DICT = {
    'factoidQA': os.path.join(os.getcwd(), "..", "dataset", "factoid_qa_dataset.jsonl"),
    'optionalQA': os.path.join(os.getcwd(), "..", "dataset", "multiple_choice_qa_dataset.jsonl"),
}

random.seed(42)

class Evaluator():
    def __init__(self, 
    ):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def compute_eval(self,
        cand, 
        ref,
        metric,
    ):
        metric_dict = {}
        for idx, (_cand, _ref) in enumerate(zip(cand, ref)):
            if _cand == '':
                _cand = 'unknown'

            if 'bert_score' == metric.name:
                _metric = metric.compute(
                    predictions=[_cand], 
                    references=[_ref],
                    lang="en"
                )
            else:
                _metric = metric.compute(
                    predictions=[_cand], 
                    references=[_ref]
                )

            if idx == 0:
                for key in _metric.keys():
                    metric_dict[key] = []

            for key in _metric.keys():
                metric_dict[key] += [_metric[key]]

        result_metric_dict = {}
        for key in metric_dict.keys():
            if isinstance(metric_dict[key][0], float) or isinstance(metric_dict[key][0], int):
                result_metric_dict[key] = float(sum(metric_dict[key]) / len(metric_dict[key]))

        return result_metric_dict

    def calulate_accuracy(self,
        cand, 
        ref,
    ):
        acc_list = []
        for pred, gt in zip(cand, ref):
            if pred == '':
                pred = 'unknown'  

            if gt in pred:
                acc_list += [1]
            else:
                acc_list += [0]

        return sum(acc_list) / len(acc_list)

    def evaluate(self,
        result_dict, 
    ):
        metrics = {}
        metrics.update(self.compute_eval(
            cand=result_dict['cand'],
            ref=result_dict['ref'],
            metric=self.bleu,
        ))
        metrics.update(self.compute_eval(
            cand=result_dict['cand'],
            ref=result_dict['ref'],
            metric=self.rouge,
        ))
        metrics.update(self.compute_eval(
            cand=result_dict['cand'],
            ref=result_dict['ref'],
            metric=self.bertscore,
        ))
        metrics['hit_rate'] = sum(result_dict['hit_ref']) / len(result_dict['hit_ref'])
        metrics['backHit_rate'] = sum(result_dict['backHit_ref']) / len(result_dict['backHit_ref'])

        if 'dataset_type' == 'optionalQA':
            metrics['accuracy'] = self.calulate_accuracy(
                cand=result_dict['cand'],
                ref=result_dict['ref'],
            )
        
        return metrics

def load_jsonl2list(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data = json.loads(line)
                data_list += [data]

    print(len(data_list), data_list[0].keys())
    return data_list

def custom_factoidQA(
    dataset_path
):
    data_path = DATASET_PATH_DICT['factoidQA'] if dataset_path is None else dataset_path
    data_list = load_jsonl2list(data_path)
    del data_list[50]
    return data_list
    
def custom_optionalQA(
    dataset_path
):
    data_path = DATASET_PATH_DICT['optionalQA'] if dataset_path is None else dataset_path
    _data_list = load_jsonl2list(data_path)

    random.shuffle(_data_list)
    _data_list = _data_list[:200]

    map_dict = {
        "? (A) ": "? Available options:\n    (A) ", 
        " (B) ": "\n    (B) ", 
        " (C) ": "\n    (C) ",
        " (D) ": "\n    (D) ",
    }

    data_list = []
    for sample in _data_list:
        for key, value in map_dict.items():
            question = sample['question'].replace(key, value)
        data_list += [{
            'question': question,
            'answer': sample['answer'],
            'url': sample['chunk_url']
        }]
    return data_list

class batchTestRunner():
    def __init__(self, 
        rag, 
        dataset_path, 
        dataset_type, 
        TOP_K=10, 
        USE_UPPER_TEXT=False, 
        USE_PRE_ANSWER=False,
    ):
        self.rag = rag
        self.dataset_type = dataset_type
        if self.dataset_type == 'factoidQA':
            self.data_list = custom_factoidQA(dataset_path)
        elif self.dataset_type == 'optionalQA':
            self.data_list = custom_optionalQA(dataset_path)

        self.TOP_K = TOP_K
        self.USE_UPPER_TEXT = USE_UPPER_TEXT
        self.USE_PRE_ANSWER = USE_PRE_ANSWER

    def ask(self,
    ):
        url_acc_list = []
        con_acc_list = []
        cand = []
        ref = []
        for idx, sample in tqdm(enumerate(self.data_list), total=len(self.data_list)):

            response = self.rag.ask(
                sample['question'], 
                top_k=TOP_K, 
                use_upper_text=USE_UPPER_TEXT, 
                pre_answer=USE_PRE_ANSWER, 
            )

            url_pred_list = []
            doc_pred_list = []
            for doc in response['relevant_docs']:
                url_pred_list += [doc.url]
                if USE_UPPER_TEXT:
                    doc_pred_list += [doc.upper_text]
                else:
                    doc_pred_list += [doc.content]
            url_gt = sample['url']

            ans_pred = response['response']
            ans_gt = sample['answer']

            if url_gt in url_pred_list:
                url_acc_list += [1]
            else:
                url_acc_list += [0]

            hit_list = []
            for content in doc_pred_list:
                if ans_gt in content:
                    hit_list += [1]
                else:
                    hit_list += [0]
            con_acc_list += [sum(hit_list) / len(hit_list)]

            cand += [ans_pred.split('Answer:')[-1]]
            ref += [ans_gt]

        return {
            'dataset_type': self.dataset_type,
            'cand': cand, 
            'ref': ref, 
            'hit_ref': url_acc_list, 
            'backHit_ref': con_acc_list
        }










