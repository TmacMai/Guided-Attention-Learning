import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model_guided_new import DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base", )
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=50)  
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument('--inter_dim', default=100, help='dimension of inter layers', type=int)   
parser.add_argument("--drop_prob", help='drop probability for dropout -- encoder', default=0.3,
                    type=float)  
parser.add_argument('--beta_shift', default=1.0, help='coefficient -- shift', type=float)
parser.add_argument('--share_dim', default=100, type=int)
parser.add_argument('--pretrained_epoch', default=5, type=int)
parser.add_argument('--cc_margin', default=1, type=float)
parser.add_argument('--cs_margin', default=0.2, type=float)
parser.add_argument('--lambda2', default=0.95, type=float)
parser.add_argument('--intensity_dim', default=8, type=int)
parser.add_argument('--ratio', default=0.2, type=float)
parser.add_argument('--loss_t_ratio', default=0.1, type=float)
parser.add_argument('--loss_p_ratio', default=0.005, type=float)  
parser.add_argument('--loss_i_ratio', default=0.005, type=float) 
parser.add_argument('--dropout_unimodal', default=0.3, type=float) 
parser.add_argument('--transformer_head', default=5, type=int) 
parser.add_argument('--transformer_layer', default=3, type=int) 
parser.add_argument('--kernel_size', default=3, type=int) 
parser.add_argument('--pretrain_iterations', default=300, type=int) 

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM,
                                      global_configs.TEXT_DIM)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id, sample_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sample_id = sample_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                sample_id = segment,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last = True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed: {}".format(seed))




def prep_for_training(num_train_optimization_steps: int):
    model = DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler, epoch_i):
    model.train()
    tr_loss = 0
    feature = False
    total_loss_p, total_loss_i, total_loss_r = [], [], []
    nb_tr_steps = 0
    if args.pretrained_epoch <= epoch_i:
        feature = True
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())
        logits, p_loss, i_loss, r_loss, loss_unimodal, (av_matrix_anoise, al_matrix_lnoise, vl_matrix_vnoise), (av_matrix, al_matrix, vl_matrix), (weight_av, weight_al, weight_vl) = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
            feature = feature
        )  ###p_loss: PAM loss, i_loss: noise loss, r_loss: enhance loss
        loss_fct = MSELoss()
        nb_tr_steps += 1
        if args.pretrained_epoch > epoch_i:
            loss =  loss_fct(logits.view(-1), label_ids.view(-1)) + args.loss_i_ratio * i_loss +  args.loss_t_ratio * r_loss  + 1 * loss_unimodal  
        else:
            loss = loss_fct(logits.view(-1), label_ids.view(-1)) + args.loss_i_ratio * i_loss +  args.loss_t_ratio * r_loss + args.loss_p_ratio * p_loss + loss_unimodal

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += i_loss.item()

        if args.pretrained_epoch <= epoch_i:
            total_loss_p.append(p_loss.item())
        total_loss_i.append(i_loss.item())
        total_loss_r.append(r_loss.item())

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


    return tr_loss / nb_tr_steps, total_loss_p, total_loss_i, total_loss_r


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, p_loss, i_loss, t_loss, loss_unimodal, (av_matrix, al_matrix, vl_matrix), (attend_av, attend_al, attend_vl, attend_noise), (weight_av, weight_al, weight_vl) = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids
            )
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, p_loss, i_loss, t_loss, loss_unimodal, (av_matrix, al_matrix, vl_matrix), (attend_av, attend_al, attend_vl, attend_noise), (weight_av, weight_al, weight_vl) = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)
    return preds, labels

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)

    mae_non = np.mean(np.absolute(preds - y_test))
    corr_non = np.corrcoef(preds, y_test)[0][1]

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, corr, f_score, mult_a7, mae_non, corr_non


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
):
  
    best_valid_loss = 1e10
    total_p, total_i, total_r =  [], [], []
    for epoch_i in range(int(args.n_epochs)):
        train_loss, total_loss_p, total_loss_i, total_loss_r = train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i)
        valid_loss = eval_epoch(model, validation_dataloader)

        total_i += total_loss_i
        total_r += total_loss_r
        if args.pretrained_epoch <= epoch_i:   
            total_p += total_loss_p

        print(
            "TRAIN: epoch:{}, train_loss:{}, valid_loss:{}".format(
                epoch_i + 1, train_loss, valid_loss
            )
        )

        if best_valid_loss > valid_loss:

            test_acc, test_mae, test_corr, test_f_score, mult_a7, test_mae_non, test_corr_non = test_score_model(
                model, test_data_loader
            )
            print(
                "TEST: train_loss:{}, valid_loss:{}, test_acc:{}, mae:{}, corr:{}, f1_score:{}, mult_a7:{}, mae_non:{}, corr_non:{}".format(
                    train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, mult_a7, test_mae_non, test_corr_non
                )
            )
            best_valid_loss = valid_loss
            best_acc = test_acc
            best_mae_non = test_mae_non
            best_corr_non = test_corr_non
            best_f_score = test_f_score
            best_acc7 = mult_a7
        print(
                "BEST TEST: best_acc:{}, best_mae_non:{}, best_corr_non:{}, best_f1_score:{}, best_acc7:{}".format(
                    best_acc, best_mae_non, best_corr_non, best_f_score, best_acc7
                )
           )
    np.save('total_p.npy', np.array(total_p))
    np.save('total_i.npy', np.array(total_i))
    np.save('total_r.npy', np.array(total_r))
    return train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score


def main():
    set_random_seed(args.seed)
    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == '__main__':
    main()
