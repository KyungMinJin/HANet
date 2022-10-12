import torch.optim as optim
from lib.core.trainer import Trainer
from lib.core.loss import HANetLoss
from lib.core.config import parse_args
from lib.models.HANet import HANet
from lib.utils.utils import create_logger, prepare_output_dir, worker_init_fn
from lib.dataset import find_dataset_using_name
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pprint
import torch
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment is {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg.GPUS)))
    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    train_dataset = dataset_class(cfg,
                                  estimator=cfg.ESTIMATOR,
                                  return_type=cfg.BODY_REPRESENTATION,
                                  phase='train')

    test_dataset = dataset_class(cfg,
                                 estimator=cfg.ESTIMATOR,
                                 return_type=cfg.BODY_REPRESENTATION,
                                 phase='test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS_NUM,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS_NUM,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # # ========= Compile Loss ========= #
    loss = HANetLoss(w_decoder=cfg.LOSS.W_DECODER,
                     lamada=cfg.LOSS.LAMADA,
                     smpl_model_dir=cfg.SMPL_MODEL_DIR,
                     smpl=(cfg.BODY_REPRESENTATION == "smpl"))

    # # ========= Initialize networks ========= #
    print(f'Slide window: {cfg.MODEL.SLIDE_WINDOW_SIZE}')
    print(f'Sample interval: {cfg.SAMPLE_INTERVAL}')
    model = HANet(cfg, test_dataset.input_dimension,
                  slide_window=cfg.MODEL.SLIDE_WINDOW_SIZE,
                  sample_interval=cfg.SAMPLE_INTERVAL,
                  encoder_hidden_dim=cfg.MODEL.ENCODER_EMBEDDING_DIMENSION,
                  decoder_hidden_dim=cfg.MODEL.DECODER_EMBEDDING_DIMENSION,
                  dropout=cfg.MODEL.DROPOUT,
                  nheads=cfg.MODEL.ENCODER_HEAD,
                  dim_feedforward=cfg.MODEL.ENCODER_EMBEDDING_DIMENSION*4,
                  enc_layers=cfg.MODEL.ENCODER_TRANSFORMER_BLOCK,
                  dec_layers=cfg.MODEL.ENCODER_TRANSFORMER_BLOCK,
                  activation="leaky_relu",
                  pre_norm=cfg.TRAIN.PRE_NORM,
                  hierarchical_encoder_interp_method='linear',
                  hierarchical_encoder_mode='transformer').to(cfg.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, amsgrad=True)

    # ========= Start Training ========= #
    Trainer(train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=model,
            loss=loss,
            writer=writer,
            optimizer=optimizer,
            cfg=cfg).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
