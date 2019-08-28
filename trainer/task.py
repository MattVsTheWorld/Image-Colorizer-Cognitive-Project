from trainer.train import run
import argparse
import logging
import os
import sys

from datetime import datetime
import tensorflow as tf


def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """

    args_parser = argparse.ArgumentParser()

    # Data files arguments
    # args_parser.add_argument(
    #     '--train-files',
    #     help='GCS or local paths to training data',
    #     nargs='+',
    #     required=True)
    # args_parser.add_argument(
    #     '--eval-files',
    #     help='GCS or local paths to evaluation data',
    #     nargs='+',
    #     required=True)
    #
    # # Experiment arguments
    # args_parser.add_argument(
    #     '--train-steps',
    #     help="""
    #   Steps to run the training job for.
    #   If --num-epochs and --train-size are not specified, this must be.
    #   Otherwise the training job will run indefinitely.
    #   if --num-epochs and --train-size are specified,
    #   then --train-steps will be: (train-size/train-batch-size) * num-epochs
    #   """,
    #     default=0,
    #     type=int)
    # args_parser.add_argument(
    #     '--eval-steps',
    #     help="""
    #   Number of steps to run evaluation for at each checkpoint.',
    #   Set to None to evaluate on the whole evaluation data.
    #   """,
    #     default=None,
    #     type=int)
    # args_parser.add_argument(
    #     '--batch-size',
    #     help='Batch size for each training and evaluation step.',
    #     type=int,
    #     default=128)
    # args_parser.add_argument(
    #     '--train-size',
    #     help='Size of training set (instance count)',
    #     type=int,
    #     default=None)
    # args_parser.add_argument(
    #     '--num-epochs',
    #     help="""\
    #     Maximum number of training data epochs on which to train.
    #     If both --train-size and --num-epochs are specified,
    #     --train-steps will be: (train-size/train-batch-size) * num-epochs.\
    #     """,
    #     default=100,
    #     type=int,
    # )
    # args_parser.add_argument(
    #     '--eval-frequency-secs',
    #     help='How many seconds to wait before running the next evaluation.',
    #     default=15,
    #     type=int)
    #
    # # Feature columns arguments
    # args_parser.add_argument(
    #     '--embed-categorical-columns',
    #     help="""
    #   If set to True, the categorical columns will be embedded
    #   and used in the deep part og the model.
    #   The embedding size = sqrt(vocab_size).
    #   """,
    #     action='store_true',
    #     default=True,
    # )
    # args_parser.add_argument(
    #     '--use-indicator-columns',
    #     help="""
    #   If set to True, the categorical columns will be encoded
    #   as One-Hot indicators in the deep part of the model.
    #   """,
    #     action='store_true',
    #     default=False,
    # )
    # args_parser.add_argument(
    #     '--use-wide-columns',
    #     help="""
    #   If set to True, the categorical columns will be used in the
    #   wide part of the model.
    #   """,
    #     action='store_true',
    #     default=False,
    # )
    #
    # # Estimator arguments
    # args_parser.add_argument(
    #     '--learning-rate',
    #     help='Learning rate value for the optimizers.',
    #     default=0.1,
    #     type=float)
    # args_parser.add_argument(
    #     '--learning-rate-decay-factor',
    #     help="""
    #   The factor by which the learning rate should decay by the end of the
    #   training.
    #   decayed_learning_rate = learning_rate * decay_rate ^ (global_step /
    #   decay_steps).
    #   If set to 1.0 (default), then no decay will occur.
    #   If set to 0.5, then the learning rate should reach 0.5 of its original
    #       value at the end of the training.
    #   Note that decay_steps is set to train_steps.
    #   """,
    #     default=1.0,
    #     type=float)
    # args_parser.add_argument(
    #     '--hidden-units',
    #     help="""
    #   Hidden layer sizes to use for DNN feature columns, provided in
    #   comma-separated layers.
    #   If --scale-factor > 0, then only the size of the first layer will be
    #   used to compute
    #   the sizes of subsequent layers.
    #   """,
    #     default='30,30,30')
    # args_parser.add_argument(
    #     '--layer-sizes-scale-factor',
    #     help="""
    #   Determine how the size of the layers in the DNN decays.
    #   If value = 0 then the provided --hidden-units will be taken as is
    #   """,
    #     default=0.7,
    #     type=float)
    # args_parser.add_argument(
    #     '--num-layers',
    #     help='Number of layers in the DNN. If --scale-factor > 0, then this '
    #          'parameter is ignored',
    #     default=4,
    #     type=int)
    # args_parser.add_argument(
    #     '--dropout-prob',
    #     help='The probability we will drop out a given coordinate.',
    #     default=None)

    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True)
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""
      Flag to decide if the model checkpoint should be re-used from the job-dir.
      If set to False then the job-dir will be deleted.
      """)
    # args_parser.add_argument(
    #     '--serving-export-format',
    #     help='The input format of the exported serving SavedModel.',
    #     choices=['JSON', 'CSV', 'EXAMPLE'],
    #     default='JSON')
    # args_parser.add_argument(
    #     '--eval-export-format',
    #     help='The input format of the exported evaluating SavedModel.',
    #     choices=['CSV', 'EXAMPLE'],
    #     default='CSV')

    return args_parser.parse_args()


def main():
    args = get_args()
    run()


if __name__ == '__main__':

    main()


'''
Cambiare il nome del job aumentando di uno il numero e il path alla cartella trainer e al file yaml
gcloud ai-platform jobs submit training madonna3 --job-dir gs://images_data/cs-job-dir/ --runtime-version 1.13 --module-name trainer.task --package-path C:\Users\Matt\Desktop\Cognitive-Project\trainer --region us-central1 --config=C:\Users\Matt\Desktop\Cognitive-Project\trainer\cloudml-gpu.yaml --stream-logs
'''
