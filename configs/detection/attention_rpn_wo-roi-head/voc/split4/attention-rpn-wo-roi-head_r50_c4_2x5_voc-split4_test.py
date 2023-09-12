_base_ = [
    '../../../_base_/datasets/query_aware/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../attention-rpn_wo-roi-head_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
num_support_ways = 2
num_support_shots = 5
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(classes='ALL_CLASSES_SPLIT4')),
    val=dict(classes='ALL_CLASSES_SPLIT4'),
    test=dict(classes='ALL_CLASSES_SPLIT4'),
    model_init=dict(classes='ALL_CLASSES_SPLIT4'))
optimizer = dict(
    lr=0.004,
    # lr = 0.0015,  # 3 * lr_default / 8
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
lr_config = dict(warmup_iters=500, warmup_ratio=0.1, step=[16000])
# runner = dict(max_iters=18000)
# runner = dict(max_iters=48000)
runner = dict(max_iters=200000)
evaluation = dict(interval=6000,
                  class_splits=['BASE_CLASSES_SPLIT4', 'NOVEL_CLASSES_SPLIT4'])
checkpoint_config = dict(interval=6000)

model = dict(
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
    ),
)
