import torch
import torch.nn as nn

from .id_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .circle_loss import CircleLoss
from .cosine_loss import CosFace, AdaCos, ArcFace
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet
from .smooth_ap_loss import SmoothAP

def build_loss_fn(cfg, num_classes):
    criterion = {}

    # TODO: convert cfg.BASELINE.COSINE_LOSS into cfg.ID_LOSS.NAME
    if cfg.MODEL.ID_LOSS.NAME == 'none':
        def id_loss_fn(score, target):
            return None
    else:
        id_loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes)
    criterion['id'] = id_loss_fn

    # !
    # TODO: add configs/
    # cfg.SOLVER.METRIC_LOSS.MARGIN
    # cfg.SOLVER.METRIC_LOSS.SCALE
    # cfg.MODEL.SOLVER.BATCH_SIZE
    # cfg.DATALOADER.NUM_INSTANCE
    if cfg.SOLVER.METRIC_LOSS.NAME == 'triplet':
        metric_loss_fn = TripletLoss(margin=cfg.SOLVER.METRIC_LOSS.MARGIN)
    # TODO: convert cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET into cfg.SOLVER.METRIC_LOSS.NAME
    elif cfg.SOLVER.METRIC_LOSS.NAME == 'weighted_regularized_triplet':
        metric_loss_fn = WeightedRegularizedTriplet()
    elif cfg.SOLVER.METRIC_LOSS.NAME == 'circle':
        metric_loss_fn = CircleLoss(m=cfg.SOLVER.METRIC_LOSS.MARGIN, s=cfg.SOLVER.METRIC_LOSS.SCALE)
    elif cfg.SOLVER.METRIC_LOSS.NAME == 'smoothAP':
        assert(cfg.SOLVER.BATCH_SIZE % cfg.DATALOADER.NUM_INSTANCE == 0)
        metric_loss_fn = SmoothAP(anneal=0.01, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                num_id=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, feat_dims=2048
        )
    else:
        def metric_loss_fn(feat, target, feat_t, target_t):
            return None
    criterion['metric'] = metric_loss_fn

    if cfg.SOLVER.CENTER_LOSS.USE:
        criterion['center'] = CenterLoss(num_classes, feat_dim=cfg.SOLVER.CENTER_LOSS.NUM_FEATS,
                                        use_gpu=True if cfg.MODEL.DEVICE = "cuda" else False
        )
    def criterion_total(score, feat, target)::
        loss = criterion['id'](score, target) + criterion['metric'](feat, target)[0]
        if cfg.SOLVER.CENTER_LOSS.USE:
            loss = loss + cfg.SOLVER.CENTER_LOSS.WEIGHT * criterion['center'](feat, target)
        return loss

    criterion['total'] = criterion_total

    return criterion
