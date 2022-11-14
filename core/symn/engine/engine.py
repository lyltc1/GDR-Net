import logging
import os
import os.path as osp
import time
import mmcv
from pytorch_lightning.lite import LightningLite
from core.symn.datasets.BOPDataset_utils import *
from core.utils import solver_utils
from lib.utils.utils import dprint
from detectron2.checkpoint import PeriodicCheckpointer
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_writer import MyCommonMetricPrinter, MyJSONWriter, MyTensorboardXWriter
from detectron2.utils.events import EventStorage
logger = logging.getLogger(__name__)


class SYMN_Lite(LightningLite):
    def get_tbx_event_writer(self, out_dir, backup=False):
        tb_logdir = osp.join(out_dir, "tb")
        mmcv.mkdir_or_exist(tb_logdir)
        if backup and self.is_global_zero:
            old_tb_logdir = osp.join(out_dir, "tb_old")
            mmcv.mkdir_or_exist(old_tb_logdir)
            os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))
        tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
        return tbx_event_writer


    def do_test(self, cfg, model, epoch=None, iteration=None):
        pass


    def do_train(self, cfg, model, optimizer, resume=False):
        model.train()
        # ---- data_loader for cfg.DATASETS.TRAIN ----
        train_dset_names = cfg.DATASETS.TRAIN
        dataset = build_train_BOP_instance_dataset(cfg, train_dset_names)
        data_loader = build_train_dataloader(cfg, dataset)
        data_loader_iter = iter(data_loader)
        # ---- data_loader for cfg.DATASETS.TRAIN2 ----
        train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
        train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
        if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
            dataset_2 = build_train_BOP_instance_dataset(cfg, train_2_dset_names)
            data_loader_2 = build_train_dataloader(cfg, dataset_2)
            data_loader_2_iter = iter(data_loader_2)
        else:
            data_loader_2 = None
            data_loader_2_iter = None
        # ---- info print ----
        total_batch_size = cfg.TRAIN.TOTAL_BATCH_SIZE
        dataset_len = len(data_loader.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset)
        iters_per_epoch = dataset_len // total_batch_size
        max_iter = cfg.TRAIN.TOTAL_EPOCHS * iters_per_epoch
        dprint("total_batch_size: ", total_batch_size)
        dprint("dataset length: ", dataset_len)
        dprint("iters per epoch: ", iters_per_epoch)
        dprint("total iters: ", max_iter)
        self.setup_dataloaders(data_loader, replace_sampler=False, move_to_device=False)
        if data_loader_2 is not None:
            self.setup_dataloaders(data_loader_2, replace_sampler=False, move_to_device=False)
        scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter)

        # resume or load model ===================================
        extra_ckpt_dict = dict(
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if hasattr(self._precision_plugin, "scaler"):
            extra_ckpt_dict["gradscaler"] = self._precision_plugin.scaler

        checkpointer = MyCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=self.is_global_zero,
            **extra_ckpt_dict,
        )
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
        if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
        else:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.SOLVER.MAX_TO_KEEP
        )

        # build writers ==============================================
        tbx_event_writer = self.get_tbx_event_writer(cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False))
        tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
        writers = (
            [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")), tbx_event_writer]
            if self.is_global_zero
            else []
        )

        logger.info("Starting training from iteration {}".format(start_iter))
        iter_time = None
        with EventStorage(start_iter) as storage:
            for iteration in range(start_iter, max_iter):
                storage.iter = iteration
                epoch = iteration // dataset_len + 1

                if np.random.rand() < train_2_ratio:
                    data = next(data_loader_2_iter)
                else:
                    data = next(data_loader_iter)

                if iter_time is not None:
                    storage.put_scalar("time", time.perf_counter() - iter_time)
                iter_time = time.perf_counter()

                # forward ============================================================
                data = batch_data_train(data)

                out_dict, loss_dict = model(
                    data["rgb_crop"],
                    obj_idx=data["obj_idx"],
                    K=data["K_crop"],
                    AABB=data["AABB_crop"],
                    gt_visib_mask=data["mask_visib_crop"],
                    gt_binary_code=data["code_crop"],
                    gt_allo_rot6d=data["allo_rot6d"],
                    gt_SITE=data["SITE"],
                    gt_R=data["cam_R_obj"],
                    gt_t=data["cam_t_obj"],
                    points=data["points"],
                    extents=data["extent"],
                    sym_infos=data["sym_info"],
                    do_loss=True,
                )
                # losses = sum(loss_dict.values())
                # assert torch.isfinite(losses).all(), loss_dict
                #
                # loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                # if self.is_global_zero:
                #     storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                #
                # optimizer.zero_grad(set_to_none=True)
                # self.backward(losses)
                # optimizer.step()
                #
                # storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                # scheduler.step()
                #
                # if (
                #         cfg.TEST.EVAL_PERIOD > 0
                #         and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                #         and iteration != max_iter - 1
                # ):
                #     self.do_test(cfg, model, epoch=epoch, iteration=iteration)
                #     # Compared to "train_net.py", the test results are not dumped to EventStorage
                #     self.barrier()
                #
                # if iteration - start_iter > 5 and (
                #         (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0 or iteration == max_iter - 1 or iteration < 100
                # ):
                #     for writer in writers:
                #         writer.write()
                #     # visualize some images ========================================
                #     if cfg.TRAIN.VIS_IMG:
                #         with torch.no_grad():
                #             vis_i = 0
                #             roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                #             roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")
                #             tbx_writer.add_image("input_image", roi_img_vis, iteration)
                #
                #             out_coor_x = out_dict["coor_x"].detach()
                #             out_coor_y = out_dict["coor_y"].detach()
                #             out_coor_z = out_dict["coor_z"].detach()
                #             out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
                #
                #             out_xyz_vis = out_xyz[vis_i].cpu().numpy().transpose(1, 2, 0)
                #             out_xyz_vis = get_emb_show(out_xyz_vis)
                #             tbx_writer.add_image("out_xyz", out_xyz_vis, iteration)
                #
                #             gt_xyz_vis = batch["roi_xyz"][vis_i].cpu().numpy().transpose(1, 2, 0)
                #             gt_xyz_vis = get_emb_show(gt_xyz_vis)
                #             tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration)
                #
                #             out_mask = out_dict["mask"].detach()
                #             out_mask = get_out_mask(cfg, out_mask)
                #             out_mask_vis = out_mask[vis_i, 0].cpu().numpy()
                #             tbx_writer.add_image("out_mask", out_mask_vis, iteration)
                #
                #             gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                #             tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)
                #
                # if (iteration + 1) % periodic_checkpointer.period == 0 or (
                #         periodic_checkpointer.max_iter is not None and (iteration + 1) >= periodic_checkpointer.max_iter
                # ):
                #     if hasattr(optimizer, "consolidate_state_dict"):  # for ddp_sharded
                #         optimizer.consolidate_state_dict()
                # periodic_checkpointer.step(iteration, epoch=epoch)
