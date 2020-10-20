# encoding: utf-8

from .baseline import Baseline
from .mgn import MGN

def build_model(cfg, num_classes):
    if cfg.MODEL.ARCHITECTURE_NAME == 'baseline':
        model = Baseline(num_classes=num_classes, 
                        last_stride=cfg.MODEL.LAST_STRIDE, 
                        model_path=cfg.MODEL.PRETRAIN_PATH,
                        backbone=cfg.MODEL.BACKBONE,
                        pool_type=cfg.MODEL.BASELINE.POOL_TYPE,
                        use_dropout=cfg.MODEL.USE_DROPOUT,
                        cosine_loss_type=cfg.MODEL.BASELINE.COSINE_LOSS_TYPE,
                        s=cfg.MODEL.BASELINE.SCALING_FACTOR,
                        m=cfg.MODEL.BASELINE.MARGIN,
                        use_bnbias=cfg.MODEL.BASELINE.USE_BNBIAS,
                        use_sestn=cfg.MODEL.BASELINE.USE_SESTN,
                        pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE
        )
    elif cfg.MODEL.ARCHITECTURE_NAME == 'mgn':
        # ! part_pool_type, use_center
        model = MGN(num_classes=num_classes,
                    last_stride=cfg.MODEL.LAST_STRIDE, 
                    model_path=cfg.MODEL.PRETRAIN_PATH,
                    backbone=cfg.MODEL.BACKBONE,
                    pool_type=cfg.MODEL.MGN.POOL_TYPE,
                    use_bnbias=cfg.MODEL.MGN.USE_BNBIAS,
                    part_pool_type=cfg.MODEL.MGN.PART_POOL_TYPE,
                    use_center=cfg.SOLVER.CENTER_LOSS.USE,
                    num_share_layer3=cfg.MODEL.MGN.NUM_SHARE_LAYER3,
        )
    else:
        print('Not found this architecture name')
    
    return model
