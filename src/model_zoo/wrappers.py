import torch
import torch.functional as F


def get_wrappers(names):
    wrappers = []
    for name in names:
        if "rcnn" in name:
            wrappers.append(RCNNEnsemble)
        elif "cascade" in name:
            wrappers.append(CascadeEnsemble)
        elif "htc" in name:
            wrappers.append(HTCEnsemble)
        else:
            raise NotImplementedError

    return wrappers


class RCNNEnsemble:
    @staticmethod
    def get_boxes(model, x, rois, img_shape, scale_factor, img_meta, num_classes):
        bbox_results = model.roi_head._bbox_forward(x, rois)
        bboxes, scores = model.roi_head.bbox_head.get_bboxes(
            rois,
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None,
        )

        # Keep only desired classes
        scores = scores[:, :num_classes]

        # Keep box corresponding to most confident class
        _, det_labels = torch.max(scores, 1)

        bboxes = bboxes.view(bboxes.size(0), -1, 4)
        bboxes = torch.stack([bboxes[i, c] for i, c in enumerate(det_labels)])

        return bboxes, scores

    @staticmethod
    def get_masks(model, x, mask_rois, num_classes):
        masks = model.roi_head._mask_forward(x, mask_rois)["mask_pred"]
        masks = masks.sigmoid().cpu().numpy()[:, :num_classes]

        return masks


class CascadeEnsemble:
    @staticmethod
    def get_boxes(model, x, rois, img_shape, scale_factor, img_meta, num_classes):
        # https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139

        ms_scores = []
        for i in range(model.roi_head.num_stages):
            bbox_results = model.roi_head._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results["cls_score"])

            if i < model.roi_head.num_stages - 1:
                cls_score = bbox_results["cls_score"]
                if model.roi_head.bbox_head[i].custom_activation:
                    cls_score = model.roi_head.bbox_head[i].loss_cls.get_activation(
                        cls_score
                    )
                bbox_label = cls_score[:, :-1].argmax(dim=1)
                rois = model.roi_head.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results["bbox_pred"], img_meta[0]
                )

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bboxes, scores = model.roi_head.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results["bbox_pred"],
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None,
        )

        scores = scores[:, :num_classes]

        return bboxes, scores

    @staticmethod
    def get_masks(model, x, mask_rois, num_classes):
        masks = []
        for i in range(model.roi_head.num_stages):
            mask = model.roi_head._mask_forward(i, x, mask_rois)['mask_pred']
            mask = mask.sigmoid()[:, :num_classes]
            masks.append(mask)
        masks = torch.stack(masks)
        masks = masks.mean(0).cpu().numpy()

        return masks


class HTCEnsemble:
    @staticmethod
    def get_boxes(model, x, rois, img_shape, scale_factor, img_meta, num_classes):
        # https://github.com/open-mmlab/mmdetection/blob/a7a16afbf2a4bdb4d023094da73d325cb864838b/mmdet/models/roi_heads/htc_roi_head.py#L505
        semantic = model.roi_head.semantic_head(x)[1]

        ms_scores = []
        for i in range(model.roi_head.num_stages):
            bbox_head = model.roi_head.bbox_head[i]
            bbox_results = model.roi_head._bbox_forward(
                i, x, rois, semantic_feat=semantic
            )
            ms_scores.append(bbox_results["cls_score"])

            if i < model.roi_head.num_stages - 1:
                bbox_label = bbox_results["cls_score"].argmax(dim=1)
                rois = bbox_head.regress_by_class(
                    rois, bbox_label, bbox_results["bbox_pred"], img_meta[0]
                )

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bboxes, scores = model.roi_head.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results["bbox_pred"],
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None,
        )

        scores = scores[:, :num_classes]

        return bboxes, scores

    @staticmethod
    def get_masks(model, x, mask_rois, num_classes):
        # https://github.com/open-mmlab/mmdetection/blob/a7a16afbf2a4bdb4d023094da73d325cb864838b/mmdet/models/roi_heads/htc_roi_head.py#L592

        mask_feats = model.roi_head.mask_roi_extractor[-1](
            x[: len(model.roi_head.mask_roi_extractor[-1].featmap_strides)], mask_rois
        )

        # Semantic feats
        semantic = model.roi_head.semantic_head(x)[1]
        mask_semantic_feat = model.roi_head.semantic_roi_extractor([semantic], mask_rois)
        if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
            mask_semantic_feat = F.adaptive_avg_pool2d(
                mask_semantic_feat, mask_feats.shape[-2:]
            )
        mask_feats += mask_semantic_feat

        last_feat = None
        masks = []
        for i in range(model.roi_head.num_stages):
            mask_head = model.roi_head.mask_head[i]
            if model.roi_head.mask_info_flow:
                mask_pred, last_feat = mask_head(mask_feats, last_feat)
            else:
                mask_pred = mask_head(mask_feats)
            masks.append(mask_pred.sigmoid()[:, :num_classes])

        masks = torch.stack(masks)
        masks = masks.mean(0).cpu().numpy()

        return masks
