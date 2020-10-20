import torch
from tqdm import tqdm
from losses import CrossEntropyLabelSmooth
from utils.util import AverageMeter, calculate_acc

def train_step(X, y_true, model, criterion, optimizer, center_loss_weight=0.0):
    img = X.to(device) if torch.cuda.device_count() >= 1 else X
    target = y_true.to(y_true) if torch.cuda.device_count() >= 1 else y_true
    score, feat = model(img)
    loss  = criterion['total'](score, feat, score)
    loss.backward()
    optimizer['model'].step()

    if 'center' in optimizer.keys():
        for param in criterion['center'].parameters():
            param.grad.data *= (1. / center_loss_weight)
        optimizer['center'].step()

    # ??
    acc = (score.max(1)[1] == target).float().mean()
    return loss.item(), acc.item()

def test_step(X, y_true, model, loss_fn, optimizer):
    y_pred = model(X, training=False)
    return y_pred

# ? may not add center_loss_weight and device
# legacy : use data_loader['train'] and  data_loader['eval']
# new : use train_loader and val_loader
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
            # train
            model.train()
            = train_step()

            # eval
            model.eval()
            = test_step()
