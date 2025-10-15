import argparse
import os
import random
import time


import numpy as np
import torch
from dataloaders.data_dataloaders import DATALOADER_DICT
from metrics.metrics import compute_metrics
from metrics.metrics import tensor_text_to_video_metrics
from metrics.metrics import tensor_video_to_text_sim

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_siglip import Siglip4IDC
from modules.optimization import BertAdam

from utils.utils import get_logger

from transformers import AutoTokenizer


global logger


def get_args(description="Siglip4IDC on Retrieval Task"):
    parser = argparse.ArgumentParser(description)

    parser.add_argument("--data_path", type=str, default="None", help="Images folder path.")
    parser.add_argument("--features_path", type=str, default="None", help="Captions feature path.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="batch size.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size eval.")
    parser.add_argument("--batch_size_val", type=int, default=64, help="batch size eval.")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay.")
    parser.add_argument("--n_display", type=int, default=100, help="Information display frequency.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument("--max_words", type=int, default=20, help="")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Decoder module.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")

    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for E.g., 0.1=10%% of training.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point to dataset to finetune.")
    parser.add_argument("--coef_lr", type=float, default=1.0, help="coefficient for bert branch.")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "" f"Invalid gradient_accumlation_steps parameter: {args.gradient_accumulation_steps}, shold be >= 1"
        )

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    return args


def init_device(args):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def init_model(args, device):
    model = Siglip4IDC.from_pretrained(pretrained_model_path=args.init_model)
    model.siglip.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, coef_lr=1.0):
    # param_optimizer = list(model.named_parameters())
    weight_decay = 0.2

    scheduler = None
    optimizer = BertAdam(
        model.siglip.parameters(),
        lr=args.lr,
        warmup=args.warmup_proportion,
        schedule="warmup_cosine",
        b1=0.9,
        b2=0.98,
        e=1e-6,
        t_total=num_train_optimization_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )

    model.siglip.to(device)
    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    output_model_file = os.path.join(
        args.output_dir,
        "pytorch_model.bin.{}{}".format(
            "" if type_name == "" else type_name + ".",
            epoch,
        ),
    )

    optimizer_state_file = os.path.join(
        args.output_dir,
        "pytorch_opt.bin.{}{}".format(
            "" if type_name == "" else type_name + ".",
            epoch,
        ),
    )

    torch.save(model.siglip.state_dict(), output_model_file)
    return output_model_file


def train_epoch(
    epoch,
    args,
    model,
    train_dataloader,
    device,
    optimizer,
    scheduler,
    global_step,
):
    global logger
    torch.cuda.empty_cache()
    model.siglip.train()

    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        try:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)

            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                image_mask,
            ) = batch

            loss = model(
                input_ids,
                segment_ids,
                input_mask,
                bef_image,
                aft_image,
                image_mask,
            )

            loss.backward()

            total_loss += float(loss)

            torch.nn.utils.clip_grad_norm_(model.siglip.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            # Clamp logit scale
            torch.clamp_(model.siglip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0:
                logger.info(
                    "Epoch: " "%d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    len(train_dataloader),
                    "-".join(
                        [
                            str("%.9f" % itm)
                            for itm in sorted(
                                list(set(optimizer.get_lr())),
                            )
                        ],
                    ),
                    float(loss),
                    (time.time() - start_time) / (log_step * args.gradient_accumulation_steps),
                )

        except Exception as e:
            logger.error(f"Error at step {step}: {str(e)}")
            raise

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(
    model,
    batch_list_t,
    batch_list_v,
    batch_sequence_output_list,
    batch_visual_output_list,
):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            pair_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(
                sequence_output,
                visual_output,
                input_mask,
                pair_mask,
            )

            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device):
    model = model.to(device)

    # ############################################################################
    # below variables are used to multi-sentence retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    ##############################################################################

    multi_sentence_ = False
    cut_off_points_, sentence_num_, pair_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, "multi_sentence_per_pair") and test_dataloader.dataset.multi_sentence_per_pair:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        pair_num_ = test_dataloader.dataset.image_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per pair setting.")
        logger.warning(f"sentence num: {sentence_num_}, pair num: {pair_num_}")

    model.siglip.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_pair_num = 0

        # ----------------------------
        # 1. Cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)

            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                image_mask,
            ) = batch

            image_pair = torch.cat([bef_image, aft_image], 1)
            if multi_sentence_:
                b, *_t = image_pair.shape

                sequence_output, _ = model.get_sequence_output(input_ids, segment_ids, input_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append(
                    (
                        input_mask,
                        segment_ids,
                    ),
                )

                s_, e_ = total_pair_num, total_pair_num + b

                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    image_pair, pair_mask = (
                        image_pair[filter_inds, ...],
                        image_mask[filter_inds, ...],
                    )

                    visual_output, _ = model.get_visual_output(image_pair, pair_mask)

                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((pair_mask))
                total_pair_num += b

            print(f"{bid}/{len(test_dataloader)}\r", end="", flush=True)

        # --------------------------------------
        # 2. calcualte the similarity
        # --------------------------------------

        sim_matrix = _run_on_single_gpu(
            model,
            batch_list_t,
            batch_list_v,
            batch_sequence_output_list,
            batch_visual_output_list,
        )

        print("type " * 10)
        print(type(sim_matrix))
        # print(sim_matrix.shape)
        print("type " * 10)

        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max(
            [
                e_ - s_
                for s_, e_ in zip(
                    [0] + cut_off_points2len_[:-1],
                    cut_off_points2len_,
                )
            ]
        )
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(
                np.concatenate(
                    (
                        sim_matrix[s_:e_],
                        np.full(
                            (max_length - e_ + s_, sim_matrix.shape[1]),
                            -np.inf,
                        ),
                    ),
                    axis=0,
                ),
            )

        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".format(
                sim_matrix.shape[0],
                sim_matrix.shape[1],
                sim_matrix.shape[2],
            ),
        )

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    logger.info("Text-to-Image-Pair:")
    logger.info(
        "\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - "
        "Mean R: {:.1f}".format(
            tv_metrics["R1"],
            tv_metrics["R5"],
            tv_metrics["R10"],
            tv_metrics["MR"],
            tv_metrics["MeanR"],
        ),
    )
    logger.info("Image-Pair-to-Text:")
    logger.info(
        "\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - "
        "V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}".format(
            vt_metrics["R1"],
            vt_metrics["R5"],
            vt_metrics["R10"],
            vt_metrics["MR"],
            vt_metrics["MeanR"],
        ),
    )

    R1 = tv_metrics["R1"]
    return R1


def main():
    global logger

    args = get_args()
    args = set_seed_logger(args)
    device = init_device(args)

    tokenizer = None

    model = init_model(args, device)

    # #########################################################
    # dataloader loading
    # #########################################################

    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        test_dataloader, test_length = val_dataloader, val_length

    logger.info("***** Running test *****")
    logger.info("  Test Num examples = %d", test_length)
    logger.info("  Test Batch size = %d", args.batch_size_val)
    logger.info("  Test Num steps = %d", len(test_dataloader))
    logger.info("***** Running val *****")
    logger.info("  Test Num examples = %d", val_length)

    # ####################################################
    # train and eval
    # ####################################################

    if args.do_train:
        train_dataloader, train_length = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (
            int(len(train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps
        ) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(
            args,
            model,
            num_train_optimization_steps,
            device,
            coef_lr,
        )

        logger.info("***** Running training *****")
        logger.info("  Training Num examples = %d", train_length)
        logger.info("  Training Batch size = %d", args.batch_size)
        logger.info(
            "  Train Num steps = %d",
            num_train_optimization_steps * args.gradient_accumulation_steps,
        )

        best_score = 0.00001
        best_output_model_file = "None"
        # ################################################################
        # resume optimizer state besides loss to continue train
        # ################################################################

        resumed_epoch = 0
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            tr_loss, global_step = train_epoch(
                epoch,
                args,
                model,
                train_dataloader,
                device,
                optimizer,
                scheduler,
                global_step,
            )

            logger.info(
                "Epoch %d/%s Finished, Train Loss: %f",
                epoch + 1,
                args.epochs,
                tr_loss,
            )

            output_model_file = save_model(
                epoch,
                args,
                model,
                optimizer,
                tr_loss,
                type_name="",
            )

            R1 = eval_epoch(args, model, test_dataloader, device)
            if best_score <= R1:
                best_score = R1
            best_output_model_file = output_model_file
            logger.info(
                f"The best model is: {best_output_model_file} the R1 is: {best_score:.4f}",
            )

    elif args.do_eval:
        print("none")
        # eval_epoch(args, model, test_dataloader, device)


if __name__ == "__main__":
    main()
