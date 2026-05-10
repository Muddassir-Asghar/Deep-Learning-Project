from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import AutoImageProcessor, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from improved_model import Model
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from PIL import Image
from torchvision import transforms


logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, pixel_values, labels):
        self.pixel_values = pixel_values
        self.labels = labels


class TextDataset(Dataset):
    def __init__(self, feature_processor, args, file_type="train"):
        crop_size = (feature_processor.size["shortest_edge"], feature_processor.size["shortest_edge"])

        if file_type == "train":
            file_path = args.train_data_file
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std),
            ])
        elif file_type == "val":
            file_path = args.eval_data_file
            self.transform = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std),
            ])
        else:
            file_path = args.test_data_file
            self.transform = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std),
            ])

        image_label = []
        if args.classify_pneumonia_type:
            logger.info("loading images for pneumonia type classification")
            all_file_path = [file_path + "/PNEUMONIA"]
            for path in all_file_path:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".jpeg"):
                            image_path = os.path.join(root, file)
                            if "bacteria" in file.lower():
                                label = 0
                            elif "virus" in file.lower():
                                label = 1
                            else:
                                logger.info("error occur when loading images!")
                                raise ValueError("Unknown pneumonia type for file")
                            image_label.append([image_path, label])
        else:
            logger.info("loading images for pneumonia detection")
            all_file_path = [file_path + "/NORMAL", file_path + "/PNEUMONIA"]
            for path in all_file_path:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".jpeg"):
                            image_path = os.path.join(root, file)
                            parent = os.path.basename(root)
                            label = 1 if parent.upper() == "PNEUMONIA" else 0
                            image_label.append([image_path, label])

        random.shuffle(image_label)
        self.examples = image_label

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        features = convert_examples_to_features(self.examples[i], self.transform)
        return features.pixel_values, torch.tensor(features.labels).long()


def convert_examples_to_features(inputs, feature_processor):
    image_path, label = inputs
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    features = feature_processor(image)
    return InputFeatures(features, label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def freeze_backbone(model):
    for p in model.vit.parameters():
        p.requires_grad = False
    for p in model.cnn.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def train(args, train_dataset, model, feature_processor, eval_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0
    )

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) * 1
    args.warmup_steps = args.max_steps // 10

    no_decay = ["bias", "LayerNorm.weight"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dataloader) * args.epochs * 0.1,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.to(args.device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 1e6

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss = model(pixel_values=pixel_values, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.save_steps == 0:
                    eval_loss = evaluate(args, model, feature_processor, eval_dataset, eval_when_training=True)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  " + "*" * 20)
                        logger.info("  Best Loss:%s", round(best_loss, 4))
                        logger.info("  " + "*" * 20)
                        checkpoint_prefix = "checkpoint-best-f1"
                        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_dir = os.path.join(output_dir, "{}".format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, feature_processor, eval_dataset, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0
    )
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    num_batch = 0
    loss_sum = 0
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            loss = model(pixel_values=pixel_values, labels=labels)
            num_batch += 1
            loss_sum += loss.sum().sum().item()

    eval_loss = loss_sum / num_batch
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, feature_processor, eval_dataset, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0
    )
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            probs = model(pixel_values=pixel_values)
            preds = torch.argmax(probs, dim=1)
            y_trues += labels.tolist()
            y_preds += preds.tolist()

    model.train()
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    precision = precision_score(y_true=y_trues, y_pred=y_preds)
    auc = roc_auc_score(y_trues, y_preds, multi_class="ovr", average="weighted")

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    specificity = tn / (tn + fp)

    logger.info("***** Test results *****")
    logger.info(f"f1: {str(f1)}")
    logger.info(f"precision: {str(precision)}")
    logger.info(f"recall: {str(recall)}")
    logger.info(f"specificity: {str(specificity)}")
    logger.info(f"Acc: {str(acc)}")
    logger.info(f"AUC: {str(auc)}")


def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--train_data_file", default=None, type=str, required=False)
    ps.add_argument("--eval_data_file", default=None, type=str, required=False)
    ps.add_argument("--test_data_file", default=None, type=str, required=False)
    ps.add_argument("--pretrain_language", default="", type=str, required=False)
    ps.add_argument("--output_dir", default=None, type=str, required=False)
    ps.add_argument("--model_type", default="roberta", type=str)
    ps.add_argument("--encoder_block_size", default=512, type=int)
    ps.add_argument("--max_line_length", default=64, type=int)
    ps.add_argument("--model_name", default="model.bin", type=str)
    ps.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str)
    ps.add_argument("--model_name_or_path", default=None, type=str)
    ps.add_argument("--config_name", default="", type=str)
    ps.add_argument("--feature_processor_name", default="", type=str)
    ps.add_argument("--do_train", action="store_true")
    ps.add_argument("--do_test", action="store_true")
    ps.add_argument("--evaluate_during_training", action="store_true")
    ps.add_argument("--train_batch_size", default=16, type=int)
    ps.add_argument("--eval_batch_size", default=16, type=int)
    ps.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ps.add_argument("--learning_rate", default=1e-4, type=float)
    ps.add_argument("--weight_decay", default=0.0, type=float)
    ps.add_argument("--adam_epsilon", default=1e-8, type=float)
    ps.add_argument("--max_grad_norm", default=1.0, type=float)
    ps.add_argument("--max_steps", default=-1, type=int)
    ps.add_argument("--warmup_steps", default=0, type=int)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--epochs", type=int, default=3)
    ps.add_argument("--classify_pneumonia_type", action="store_true")
    ps.add_argument("--focal_gamma", type=float, default=2.0)
    ps.add_argument("--focal_alpha", type=float, default=-1.0)
    args = ps.parse_args()

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()
        args.device = "cuda"
    else:
        args.n_gpu = 0
        args.device = "cpu"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args)

    feature_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    vit = AutoModel.from_pretrained(args.model_name_or_path)
    model = Model(vit, feature_processor, args)

    state_dict = torch.load(
        "./saved_models/checkpoint-best-f1/domain_adapted_davit.bin", map_location=args.device
    )
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for layer_name, weights in state_dict.items():
        if layer_name in model_state_dict and model_state_dict[layer_name].shape == weights.shape:
            filtered_state_dict[layer_name] = weights
    model.load_state_dict(filtered_state_dict, strict=False)

    freeze_backbone(model)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = TextDataset(feature_processor, args, file_type="train")
        eval_dataset = TextDataset(feature_processor, args, file_type="val")
        train(args, train_dataset, model, feature_processor, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f"checkpoint-best-f1/{args.model_name}"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(feature_processor, args, file_type="test")
        test(args, model, feature_processor, test_dataset)


if __name__ == "__main__":
    main()
