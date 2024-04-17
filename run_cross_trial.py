import sys
sys.path.append("..")
from utils.dl_utils import set_random_seed
from train_test import train_test_session


def subject_cross_trial(subject, config, dataset="SEED", learn_type="SLL",
                          network="MLP", sess_ids=None):
    """
    This function runs cross-trial experiment for the given subject
    :param subject: Subject to run experiment on
    :param learn_type: Type of label assignment learning method
    :param model: type of model to be trained
    :param sess_ids: ID of the sessions to train on (available 3)
    :return: None
    """
    if sess_ids is None:
        sess_ids = [1, 3]
    train_test_session(subject, learn_type, network, sess_ids, dataset, config)


def main():
    set_random_seed(seed=42)
    subject_list = [i for i in range(1, 16)]
    learn_type = "LDL"
    network = "MLP"
    dataset = "SEED5"

    # config = {"lr": 0.0028, "weight_decay": 0.018, "bs": 1024} # SEED
    config = {"lr": 0.0012, "weight_decay": 2.58e-5, "bs": 1024} # SEED5

    for subject in subject_list:
        subject_cross_trial(subject=subject, learn_type=learn_type,
                            network=network, sess_ids=[1, 2, 3], dataset=dataset,
                            config=config)


if __name__ == '__main__':
    main()
