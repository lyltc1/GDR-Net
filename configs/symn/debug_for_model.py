MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        NAME="GDRN",
        TASK="rot",
        USE_MTL=False,  # uncertainty multi-task weighting
        BACKBONE=dict(
            PRETRAINED="torchvision://resnet34",
            ARCH="resnet",
            NUM_LAYERS=34,
            INPUT_CHANNEL=3,
            INPUT_RES=256,
            OUTPUT_RES=64,
            FREEZE=False,
        ),
        ROT_HEAD=dict(
            NUM_CLASSES=21,
            # for region classification, 0 is bg, [1, num_regions]
            # num_regions <= 1: no region classification
            NUM_REGIONS=64,
            FREEZE=False,
            ROT_CONCAT=True,
            XYZ_BIN=64,  # for classification xyz, the last one is bg
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            CONV_KERNEL_SIZE=3,
            NORM="BN",
            NUM_GN_GROUPS=32,
            OUT_CONV_KERNEL_SIZE=1,
            ROT_CLASS_AWARE=False,
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            MASK_THR_TEST=0.5,
            REGION_CLASS_AWARE=False,
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,  # using 2D XY coords
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z", #  trans | centroid_z (SITE) | centroid_z_abs
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LOSS_SYM=True,  # use symmetric PM loss
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
            FREEZE=False,
            LR_MULT=1.0,
            # ConvPnPNet | SimplePointPnPNet | PointPnPNet | ResPointPnPNet
            PNP_HEAD_CFG=dict(type="ConvPnPNet", norm="GN", num_gn_groups=32, drop_prob=0.0),  # 0.25
            # PNP_HEAD_CFG=dict(
            #     type="ConvPnPNet",
            #     norm="GN",
            #     num_gn_groups=32,
            #     spatial_pooltype="max", # max | mean | soft | topk
            #     spatial_topk=1,
            #     region_softpool=False,
            #     region_topk=8,  # NOTE: default the same as NUM_REGIONS
            # ),
            MASK_ATTENTION="none",  # none | concat | mul
            TRANS_WITH_BOX_INFO="none",  # none | ltrb | wh  # TODO
            ## for losses
            # {allo/ego}_{quat/rot6d/log_quat/lie_vec}
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG  (only valid for centroid_z)
            # point matching loss
            NUM_PM_POINTS=3000,
            PM_LOSS_TYPE="L1",  # L1 | Smooth_L1
            PM_SMOOTH_L1_BETA=1.0,
            # if False, the trans loss is in point matching loss
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=False,  # disentangle R/xy/z
            PM_T_USE_POINTS=False,
            PM_LW=1.0,
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_LW=0.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            # bind term loss: R^T@t
            BIND_LOSS_TYPE="L1",
            BIND_LW=0.0,
        ),
        TRANS_HEAD=dict(
            ENABLED=False,
            FREEZE=True,
            LR_MULT=1.0,
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            NORM="BN",
            NUM_GN_GROUPS=32,
            CONV_KERNEL_SIZE=3,
            OUT_CHANNEL=3,
            TRANS_TYPE="centroid_z",  # trans | centroid_z
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=0.0,
            Z_LOSS_TYPE="L1",
            Z_LW=0.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LW=0.0,
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)
SOLVER = dict(
    IMS_PER_BATCH=6,
    TOTAL_EPOCHS=160,
    # NOTE: use string code to get cfg dict like mmdet
    # will ignore OPTIMIZER_NAME, BASE_LR, MOMENTUM, WEIGHT_DECAY
    OPTIMIZER_CFG=dict(type="RMSprop", lr=1e-4, momentum=0.0, weight_decay=0),
    #######
    GAMMA=0.1,
    BIAS_LR_FACTOR=1.0,
    LR_SCHEDULER_NAME="WarmupMultiStepLR",  # WarmupMultiStepLR | flat_and_anneal
    WARMUP_METHOD="linear",
    WARMUP_FACTOR=1.0 / 1000,
    WARMUP_ITERS=1000,
    ANNEAL_METHOD="step",
    ANNEAL_POINT=0.75,
    POLY_POWER=0.9,  # poly power
    REL_STEPS=(0.5, 0.75),
    # checkpoint
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
    # Enable automatic mixed precision for training
    # Note that this does not change model's inference behavior.
    # To use AMP in inference, run inference under autocast()
    AMP=dict(ENABLED=False),
)