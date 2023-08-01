'''
Script for fine-tuning and evaluating the BERT classifier in distinguishing between GPT-3 generated articles and human written articles.

Relies on functions defined in modules/finetune_fns.py

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# utils 
import pathlib
import argparse

# to define parameters for model 
from transformers import TrainingArguments 

# custom modules
from modules.finetune_fns import finetune, get_loss, plot_loss, get_metrics, prepare_data

# save log history
import pickle

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-hub", "--push_to_hub", help = "Whether to push to Hugging Face Hub, if arg is specified, it will push to hub", action="store_true") 
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model should run for", type = int, default = 15)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args


def main(): 
    # intialise args 
    args = input_parse()

    # define paths (for saving model and loss curve)
    path = pathlib.Path(__file__)
    output_folder = path.parents[2] / "model"
    resultspath = path.parents[2] / "results"

    # import data
    path_train = path.parents[2] / "data" / "labelled_data_for_classifier.csv"
    path_test =  path.parents[2] / "data" / "test_data_classifier.csv"

    ds = prepare_data(path_train, path_test)

    # define model 
    model_name = "NbAiLab/nb-bert-large"

    # login for push to hub functionality! But only if "-hub" flag is specified
    if args.push_to_hub: 
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[1] / "token.txt") as f:
            hf_token = f.read()

        login(hf_token)

    # define batch_size 
    batch_size = 24

    # define training arguments 
    training_args = TrainingArguments(
        output_dir = output_folder,  
        push_to_hub = args.push_to_hub, # only if flag is specified 
        learning_rate=2e-5,
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size, 
        num_train_epochs=args.n_epochs, 
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy = "epoch", 
        load_best_model_at_end = True, 
        metric_for_best_model = "accuracy",
    )

    # fine-tune 
    trainer, tokenized_data = finetune(
        dataset = ds, 
        model_name = model_name,
        n_labels = 2,
        training_args = training_args, 
        early_stop_patience=3
        )

    # push model to hub (only if "-hub" flag is specified)
    if args.push_to_hub: 
        trainer.push_to_hub()

    # save log history with pickle
    with open (resultspath / f"{args.model}_log_history.pkl", "wb") as file:
        pickle.dump(trainer.state.log_history, file)

    # compute train and val loss, plot loss
    train_loss, val_loss, total_epochs = get_loss(trainer.state.log_history)
    plot_loss(train_loss, val_loss, total_epochs, resultspath, f"{args.model}_loss_curve.png")

if __name__ == "__main__":
    main()