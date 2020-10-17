import torch
from tqdm import tqdm
from losses import CrossEntropyLabelSmooth
from utils.util import AverageMeter, calculate_acc

def train_step(X, y_true, model, loss_fn, optimizer):
    # TODO: implement


    return loss

def test_step(X, y_true, model, loss_fn, optimizer):
    y_pred = model(X, training=False)
    return y_pred

# ? may not add center_loss_weight and device
def do_train(cfg, model, train_loader, val_loader, optimizer=None, \
            scheduler=None, loss_fn=None, ckpt_path=None, \
            center_loss_weight=0.0, device=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(lr=1e-3)

        if loss_fn is None:
            # ! cfg.DATASET.NUM_CLASSES
            loss_fn = CrossEntropyLabelSmooth(num_classes=cfg.DATASET.NUM_CLASSES)

        log_period = cfg.SOLVER.LOG_PERIOD
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        eval_period = cfg.SOLVER.EVAL_PERIOD
        output_dir = cfg.OUTPUT_DIR
        device = cfg.MODEL.DEVICE
        epochs = cfg.SOLVER.MAX_EPOCHS

        logger = logging.getLogger("reid_baseline")
        logger.info("Start training")

        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR + '/writer')

        avg_loss = AverageMeter()
        val_loss = AverageMeter()
        acc = AverageMeter()
        val_acc = AverageMeter()

        min_val_loss = np.inf

        model.train()
        optimizer['model'].zero_grad()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()

        for epoch in tqdm(range(cfg.SOLVER.MAX_EPOCHS)):
                train_loader 
