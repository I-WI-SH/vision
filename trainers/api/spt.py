import os
import os.path as osp
import numpy as np
import json
from tqdm import tqdm
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torchattacks import PGD, TPGD
from autoattack import AutoAttack
from torch.autograd import grad, Variable
import pdb

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):

    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = os.environ.get('CLIP_MODEL_PATH')
    if not model_path or not model_path.strip():
        raise ValueError('未配置 CLIP 权重路径，请设置 CLIP_MODEL_PATH 或请求中的 model_uri')
    if not osp.isfile(model_path):
        raise FileNotFoundError(f'CLIP 权重文件不存在：{model_path}')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    if not zero_shot_model:
        design_details = {"trainer": 'DEFEAT',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        # model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'DEFEAT',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        # model = clip.build_model(state_dict or model.state_dict(), design_details)
        
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    if cfg.MODEL.BACKBONE.ROBUST:
        backbone_dir = os.environ.get("APT_ROBUST_BACKBONE_DIR")

        if not backbone_dir or not backbone_dir.strip():
            raise ValueError(
                "未配置 Robust CLIP 权重目录，请设置 "
                "APT_ROBUST_BACKBONE_DIR 或请求中的 robust_backbone_dir"
            )
        if not osp.isdir(backbone_dir):
            raise FileNotFoundError(
                f"Robust CLIP 权重目录不存在：{backbone_dir}"
            )
        
        ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
        eps = int(cfg.AT.EPS * 255)
        ckp_name += f'_eps{eps}.pth.tar'
        checkpoint_path = osp.join(backbone_dir, ckp_name)
        if not osp.isfile(checkpoint_path):
            raise FileNotFoundError(f'Robust CLIP 权重文件不存在：{checkpoint_path}')
        ckp = torch.load(checkpoint_path, weights_only=False)
        missing_keys_4_robust_clip, _ = model.visual.load_state_dict(ckp['vision_encoder_state_dict'], strict=False)
        # print('Weights not found for some missing keys_for robust clip: ', missing_keys_4_robust_clip)
        print('Load Robust Clip TeCoA')
    return model

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        ##### original clip text features ############################################
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg,True)
        clip_model_.cuda()
        
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp_hard = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp_hard.format(c.replace("_", " ")) for c in classnames] #hand-cafted prompt
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_]) 
        prompts_ = prompts_.cuda()
        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_) 
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features #originial text_features
        #############################################################################

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts

class LinearProjection(nn.Module):
    def __init__(self, dim):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x
    
class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


# residual VQ-VAE
class ResidualVQVAE_NET(nn.Module):

    def __init__(self, input_dim, dim, n_embedding, num_quantizers=5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim)
        )

        self.num_quantizers = num_quantizers

        # multi codebook
        self.vq_embedding = nn.ModuleList(
            [
                nn.Embedding(n_embedding, dim)
                for _ in range(num_quantizers)
            ]
        )

        for embedding in self.vq_embedding:
            embedding.weight.data.uniform_(
                -1.0 / n_embedding,
                1.0 / n_embedding
            )


        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, input_dim, 3, 1, 1)
        )


    def quantize(self, x, embedding):

        """
        x:
        [N,C,H,W]

        embedding:
        [K,C]

        return:
        quantized feature
        """

        N,C,H,W=x.shape
        K,_=embedding.shape

        embedding_broadcast = (
            embedding
            .reshape(1,K,C,1,1)
        )

        x_broadcast = (
            x.reshape(N,1,C,H,W)
        )

        distance = torch.sum(
            (embedding_broadcast-x_broadcast)**2,
            dim=2
        )

        nearest_neighbor=torch.argmin(
            distance,
            dim=1
        )

        z_q = embedding[nearest_neighbor]

        z_q = z_q.permute(
            0,3,1,2
        )

        return z_q, nearest_neighbor


    def forward(self, x):
        # [N, 49, 512] -> [N, 512, 7, 7]
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(x.shape[0], 512, 7, 7)

        ze=self.encoder(x)

        residual_feature = ze.detach()
        aggregated_quants = torch.zeros_like(ze)

        quantized_list=[]
        indices=[]

        # Residual quantization
        for embedding_layer in self.vq_embedding:

            z_q, idx = self.quantize(
                residual_feature,
                embedding_layer.weight
            )


            residual_feature = residual_feature - z_q.detach()
            aggregated_quants = aggregated_quants + z_q

            quantized_list.append(aggregated_quants)
            indices.append(idx.unsqueeze(-1))


        zq=quantized_list[-1]

        # straight-through estimator
        decoder_input = ze + (zq-ze).detach()
        x_hat=self.decoder(decoder_input)


        return (
            x_hat,
            ze,
            zq,
            quantized_list
        )



class QuantizationPairStats:
    """Inspect the same clean/adversarial pairs, keeping sample-level evidence."""

    GROUPS = ("all_same", "90_to_100", "50_to_90", "20_to_50", "below_20")

    def __init__(self):
        self.num_pairs = 0
        self.patch_sums = {}
        self.samples = []
        self.code_counts = None

    @staticmethod
    def stability_group(same, total):
        if same == total:
            return "all_same"
        for percent, name in ((90, "90_to_100"), (50, "50_to_90"), (20, "20_to_50")):
            if same * 100 >= percent * total:
                return name
        return "below_20"

    @staticmethod
    def code_usage(counts):
        counts = np.asarray(counts, dtype=np.int64)
        total = int(counts.sum())
        active = np.flatnonzero(counts)
        probabilities = counts[active] / total if total else np.array([])
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        return {
            "num_unique_codes": int(active.size),
            "codebook_coverage": float(active.size / counts.size),
            "top1_patch_fraction": float(counts.max() / total) if total else None,
            "effective_num_codes": float(np.exp(entropy)) if total else None,
            "counts": {str(code): int(counts[code]) for code in active},
        }

    @torch.no_grad()
    def update(self, clean, adversarial, labels, clean_logits, adv_logits,
               num_codes, paths=None, extra_patch_values=None):
        ze_clean, zq_clean, codes_clean = clean
        ze_adv, zq_adv, codes_adv = adversarial
        if (ze_clean.ndim != 4 or ze_clean.shape != ze_adv.shape
                or ze_clean.shape != zq_clean.shape or ze_clean.shape != zq_adv.shape
                or codes_clean.shape != codes_adv.shape
                or codes_clean.shape != ze_clean.shape[:1] + ze_clean.shape[2:]):
            raise ValueError("Expected aligned latents [B, C, H, W] and codes [B, H, W].")
        patch_values = {
            "pre_quant_l2": (ze_clean.float() - ze_adv.float()).norm(dim=1),
            "post_quant_l2": (zq_clean.float() - zq_adv.float()).norm(dim=1),
            "same_code_rate": codes_clean.eq(codes_adv).double(),
        }
        if extra_patch_values:
            patch_values.update(extra_patch_values)
        for name, values in patch_values.items():
            total = values.sum(dim=0, dtype=torch.float64).cpu()
            if name not in self.patch_sums:
                self.patch_sums[name] = torch.zeros_like(total)
            if total.shape != self.patch_sums[name].shape:
                raise ValueError("Patch grid changed during evaluation.")
            self.patch_sums[name] += total

        self._record_samples(
            codes_clean.cpu().numpy(), codes_adv.cpu().numpy(),
            labels.cpu().tolist(), clean_logits.argmax(dim=1).cpu().tolist(),
            adv_logits.argmax(dim=1).cpu().tolist(),
            {name: values.flatten(1).mean(dim=1).cpu().tolist()
             for name, values in patch_values.items() if name != "same_code_rate"},
            num_codes, paths,
        )

    def _record_samples(self, clean_codes, adv_codes, labels, clean_preds,
                        adv_preds, distances, num_codes, paths=None):
        if clean_codes.shape != adv_codes.shape or clean_codes.ndim != 3:
            raise ValueError("Expected two aligned code grids [B, H, W].")
        batch_size = clean_codes.shape[0]
        if any(len(values) != batch_size for values in (
                labels, clean_preds, adv_preds, *distances.values())):
            raise ValueError("Predictions and features must refer to the same samples.")
        if paths is not None and len(paths) != batch_size:
            raise ValueError("Image paths must align with the batch.")
        if num_codes <= 0 or any(np.any((codes < 0) | (codes >= num_codes))
                                 for codes in (clean_codes, adv_codes)):
            raise ValueError("Code IDs must be within the codebook.")
        if self.code_counts is None:
            self.code_counts = {side: np.zeros(num_codes, dtype=np.int64)
                                for side in ("clean", "adv")}
        elif len(self.code_counts["clean"]) != num_codes:
            raise ValueError("Codebook size changed during evaluation.")

        for index, (clean_grid, adv_grid) in enumerate(zip(clean_codes, adv_codes)):
            mask = clean_grid == adv_grid
            same, total = int(mask.sum()), int(mask.size)
            clean_correct = clean_preds[index] == labels[index]
            adv_correct = adv_preds[index] == labels[index]
            row = {
                "sample_index": self.num_pairs,
                "image_path": str(paths[index]) if paths is not None else None,
                "label": int(labels[index]),
                "clean_prediction": int(clean_preds[index]),
                "adv_prediction": int(adv_preds[index]),
                "clean_correct": bool(clean_correct),
                "adv_correct": bool(adv_correct),
                "prediction_changed": bool(clean_preds[index] != adv_preds[index]),
                "transition": ("correct" if clean_correct else "wrong") + "_to_" +
                              ("correct" if adv_correct else "wrong"),
                "num_patches": total,
                "num_same_codes": same,
                "same_code_rate": same / total,
                "all_patches_same": same == total,
                "stability_group": self.stability_group(same, total),
                "same_code_mask": mask.tolist(),
                "clean_codes": clean_grid.tolist(),
                "adv_codes": adv_grid.tolist(),
                **{name: float(values[index]) for name, values in distances.items()},
            }
            for side, grid in (("clean", clean_grid), ("adv", adv_grid)):
                counts = np.bincount(grid.reshape(-1), minlength=num_codes)
                self.code_counts[side] += counts
                row[f"{side}_code_usage"] = self.code_usage(counts)
            self.samples.append(row)
            self.num_pairs += 1

    def classification_summary(self, rows):
        count = len(rows)
        clean_correct = sum(row["clean_correct"] for row in rows)
        adv_correct = sum(row["adv_correct"] for row in rows)
        transitions = {name: sum(row["transition"] == name for row in rows)
                       for name in ("correct_to_correct", "correct_to_wrong",
                                    "wrong_to_correct", "wrong_to_wrong")}
        return {
            "count": count,
            "fraction_of_dataset": count / self.num_pairs,
            "clean_accuracy": clean_correct / count if count else None,
            "adv_accuracy": adv_correct / count if count else None,
            "prediction_change_rate": (sum(row["prediction_changed"] for row in rows)
                                       / count) if count else None,
            "attack_success_rate_on_clean_correct": (
                transitions["correct_to_wrong"] / clean_correct) if clean_correct else None,
            "mean_same_code_rate": (sum(row["same_code_rate"] for row in rows)
                                    / count) if count else None,
            "transitions": transitions,
        }

    def sample_summary(self):
        if not self.num_pairs:
            raise ValueError("Cannot summarize an empty loader.")
        groups = {}
        for name in self.GROUPS:
            rows = [row for row in self.samples if row["stability_group"] == name]
            groups[name] = self.classification_summary(rows)
            # groups[name]["by_true_class"] = {
            #     str(label): self.classification_summary([row for row in rows
            #                                             if row["label"] == label])
            #     for label in sorted({row["label"] for row in rows})
            # }
        thresholds = {
            f"at_least_{percent}_percent": self.classification_summary([
                row for row in self.samples
                if row["num_same_codes"] * 100 >= percent * row["num_patches"]
            ]) for percent in (100, 90, 50, 20)
        }
        usage = {}
        for side, counts in self.code_counts.items():
            usage[side] = self.code_usage(counts)
            # usage[side]["all_code_counts"] = counts.tolist()
            for metric in ("num_unique_codes", "top1_patch_fraction", "effective_num_codes"):
                values = [row[f"{side}_code_usage"][metric] for row in self.samples]
                usage[side][f"mean_per_image_{metric}"] = float(np.mean(values))
            unique_counts = [row[f"{side}_code_usage"]["num_unique_codes"]
                             for row in self.samples]
            usage[side]["per_image_unique_codes_histogram"] = {
                str(value): unique_counts.count(value) for value in sorted(set(unique_counts))
            }
        return {
            "classification": self.classification_summary(self.samples),
            "stability_groups": groups,
            # "cumulative_thresholds": thresholds,
            "codebook_size": len(self.code_counts["clean"]),
            "code_usage": usage,
        }

    def compute(self):
        summary = self.sample_summary()
        maps = {name: total / self.num_pairs for name, total in self.patch_sums.items()}
        return {
            "num_pairs": self.num_pairs,
            "patch_grid": list(next(iter(maps.values())).shape),
            "overall": {name: values.mean().item() for name, values in maps.items()},
            # "per_patch": {name: values.tolist() for name, values in maps.items()},
            **summary,
        }


class ResidualQuantizationPairStats:
    """Reuse all single-layer diagnostics independently at every residual depth."""

    def __init__(self):
        self.layers = []

    @torch.no_grad()
    def update(self, clean, adversarial, labels, clean_logits, adv_logits,
               num_codes, paths=None):
        ze_clean, rq_clean = clean
        ze_adv, rq_adv = adversarial
        num_layers = rq_clean["codes"].size(1)
        if rq_clean["codes"].shape != rq_adv["codes"].shape:
            raise ValueError("Clean/adv RQ code grids must align.")
        if len(num_codes) != num_layers:
            raise ValueError("Expected one codebook size per layer.")
        if not self.layers:
            self.layers = [QuantizationPairStats() for _ in range(num_layers)]
        if len(self.layers) != num_layers:
            raise ValueError("RQ depth changed during evaluation.")
        prefix_same = torch.ones_like(rq_clean["codes"][:, 0], dtype=torch.bool)
        for level, stats in enumerate(self.layers):
            clean_codes = rq_clean["codes"][:, level]
            adv_codes = rq_adv["codes"][:, level]
            prefix_same = prefix_same & clean_codes.eq(adv_codes)
            cumulative_distance = (
                rq_clean["cumulative"][:, level].float()
                - rq_adv["cumulative"][:, level].float()
            ).norm(dim=1)
            old_count = stats.num_pairs
            stats.update(
                (rq_clean["residuals"][:, level], rq_clean["quantized"][:, level], clean_codes),
                (rq_adv["residuals"][:, level], rq_adv["quantized"][:, level], adv_codes),
                labels, clean_logits, adv_logits, num_codes[level], paths,
                extra_patch_values={
                    "cumulative_quant_l2": cumulative_distance,
                    "prefix_same_code_rate": prefix_same.double(),
                },
            )
            for row, mask in zip(stats.samples[old_count:], prefix_same.cpu().numpy()):
                same, total = int(mask.sum()), int(mask.size)
                row["prefix_same_code_mask"] = mask.tolist()
                row["prefix_num_same_codes"] = same
                row["prefix_all_patches_same"] = same == total
                row["prefix_stability_group"] = stats.stability_group(same, total)

    @property
    def samples(self):
        if not self.layers:
            return
        for index in range(self.layers[0].num_pairs):
            yield {
                "sample_index": index,
                "layers": {f"layer_{level + 1}": stats.samples[index]
                           for level, stats in enumerate(self.layers)},
            }

    def compute(self):
        if not self.layers:
            raise ValueError("Cannot summarize an empty loader.")
        layers = {}
        for level, stats in enumerate(self.layers):
            summary = stats.compute()
            # Classification still uses the full model, grouped by prefix stability.
            prefix_rows = [dict(row, same_code_rate=row["prefix_same_code_rate"],
                                stability_group=row["prefix_stability_group"])
                           for row in stats.samples]
            summary["prefix_stability_groups"] = {
                name: stats.classification_summary([
                    row for row in prefix_rows if row["stability_group"] == name
                ]) for name in stats.GROUPS
            }
            layers[f"layer_{level + 1}"] = summary
        first, last = layers["layer_1"], layers[f"layer_{len(self.layers)}"]
        return {
            "num_pairs": first["num_pairs"],
            "num_quantizers": len(self.layers),
            "patch_grid": first["patch_grid"],
            "overall": {
                "pre_quant_l2": first["overall"]["pre_quant_l2"],
                "post_quant_l2": last["overall"]["cumulative_quant_l2"],
                "same_code_rate": last["overall"]["prefix_same_code_rate"],
            },
            "stability_groups": last["prefix_stability_groups"],
            "layers": layers,
        }


class ImageNormalizer(nn.Module):

    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device='cuda'):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.cls_projection = LinearProjection(49).to(clip_model.dtype)
        self.ori_embedding = self.prompt_learner.text_features
        self.vqvae = ResidualVQVAE_NET(512, 256, 128, num_quantizers=5).to(clip_model.dtype) # input_dim, dim, n_embedding
        self.normalize = ImageNormalizer(cfg.INPUT.PIXEL_MEAN,
                                         cfg.INPUT.PIXEL_STD).to(device)
        self.alpha = cfg.TRAINER.SPT.ALPHA 
        self.cfg = cfg
    


    @torch.no_grad()
    def quantization_features(self, image):
        """Diagnostic-only extraction using the existing residual quantizer verbatim."""
        _, local_features = self.image_encoder(self.normalize(image).type(self.dtype))
        vq = self.vqvae
        x = local_features.permute(0, 2, 1).contiguous()
        ze = vq.encoder(x.reshape(x.shape[0], 512, 7, 7))
        residual = ze.detach()
        cumulative = torch.zeros_like(ze)
        residuals, quantized, prefixes, codes = [], [], [], []
        for embedding in vq.vq_embedding:
            residuals.append(residual)
            zq, indices = vq.quantize(residual, embedding.weight)
            residual = residual - zq.detach()
            cumulative = cumulative + zq
            quantized.append(zq)
            prefixes.append(cumulative)
            codes.append(indices)
        return ze, {
            "residuals": torch.stack(residuals, dim=1),
            "quantized": torch.stack(quantized, dim=1),
            "cumulative": torch.stack(prefixes, dim=1),
            "codes": torch.stack(codes, dim=1),
        }

    def forward(self, image, attack=False):

        image_features, local_ima_fea = self.image_encoder(self.normalize(image).type(self.dtype))

        x_hat, ze, zq, rq = self.vqvae(local_ima_fea)
        ima_fea_vae = x_hat.view(x_hat.shape[0], 512, -1)
        ima_fea_vae = ima_fea_vae.permute(0, 2, 1)

        local_ima_fea__ = self.cls_projection(ima_fea_vae)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_ima_fea__ = local_ima_fea__ / local_ima_fea__.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_ima_fea__ @ text_features.t()
        logits_mix = self.alpha * logits + (1-self.alpha) *logits_local

        if self.prompt_learner.training:
            return logits_mix, image_features, local_ima_fea__, text_features, self.ori_embedding,  x_hat, ze, zq, local_ima_fea, rq
        else:
            return logits_mix


@TRAINER_REGISTRY.register()
class SPT(TrainerX):
    def check_cfg(self, cfg):
        print(cfg.TRAINER.COOP.PREC)
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)


        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                elif "cls_projection" in name:
                    param.requires_grad_(True)
                elif "vqvae" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        trainable_params = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                enabled.add(name)
                trainable_params += param.numel()
                print(f"{name}: shape={tuple(param.shape)}, 参数量={param.numel():,}")

        print(f"Parameters to be updated: {enabled}")
        print(f"可训练参数张量数量: {len(enabled)}")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.4f} M)")
        print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.4f} M)")
        print(f"冻结参数量: {total_params - trainable_params:,}")
        if total_params > 0:
            print(f"可训练参数占比: {100 * trainable_params / total_params:.4f}%")

        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        optim = self.optim

        eps=self.cfg.AT.EPS
        alpha=self.cfg.AT.ALPHA
        steps=self.cfg.AT.STEPS
        attack = PGD(self.model,
                     eps=self.cfg.AT.EPS,
                     alpha=self.cfg.AT.ALPHA,
                     steps=self.cfg.AT.STEPS)


        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:

            delta = torch.zeros_like(image).uniform_(-eps, eps)
            for _ in range(steps):
                adv = torch.clamp(image+delta, 0, 1).requires_grad_(True)
                output, *_, = self.model(adv, attack=True)
                loss_ce = -F.cross_entropy(output, label)
                grad_adv = torch.autograd.grad(loss_ce, adv)[0]
                delta -= alpha * torch.sign(grad_adv)
                delta = torch.clamp(delta, -eps, eps).detach()
                
            adv = torch.clamp(image+delta, 0, 1).detach()
            output, image_features, local_ima_fea__, text_features, ori_embedding, x_hat, ze, zq, x, rq = self.model(adv)
            output_clean, *_ = self.model(image)

            loss_ce_adv = F.cross_entropy(output, label)
            loss_ce_clean = F.cross_entropy(output_clean, label)

            # KL(clean || adv)：clean 作为目标分布，KL 梯度仅流向 adv 分支。
            temperature = self.cfg.TRAINER.SPT.TEMPERATURE
            loss_kl = F.kl_div(
                F.log_softmax(output / temperature, dim=-1),
                F.softmax(output_clean.detach() / temperature, dim=-1),
                reduction="batchmean",
            )
            # ) * temperature**2

            loss_ce = (
                0.5 * loss_ce_adv
                + 0.5 * loss_ce_clean
                + 1.0 * loss_kl
            )      


            loss_scl_image = F.l1_loss(image_features, local_ima_fea__.cuda(),
                                reduction='mean')
            loss_scl_text = F.l1_loss(text_features, ori_embedding.cuda(),
                                      reduction='mean')

            mse_loss = nn.MSELoss()
            x = x.permute(0,2,1)
            x = x.view(x_hat.shape)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            # l_commitment = mse_loss(ze, zq.detach())

            l_commitment_list = []
            for quantized in rq:
                l_commitment_list.append(mse_loss(ze, quantized.detach()))
            l_commitment = torch.mean(torch.stack(l_commitment_list))


            loss_vae = self.cfg.TRAINER.SPT.W3 * l_reconstruct + self.cfg.TRAINER.SPT.W4 * l_embedding + 0.25 * self.cfg.TRAINER.SPT.W4 * l_commitment
            loss_scl = self.cfg.TRAINER.SPT.W1 * loss_scl_image + self.cfg.TRAINER.SPT.W2 * loss_scl_text


            loss = loss_ce + loss_scl + loss_vae 


            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss_ce": loss_ce.item(),
            "loss_scl": loss_scl.item(),
            "l_reconstruct": l_reconstruct.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)


        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def _save_quantization_stats(self, pair_stats, split):
        results = pair_stats.compute()
        results["split"] = split
        results["class_names"] = {str(key): str(value)
                                  for key, value in (self.lab2cname or {}).items()}
        results["attack"] = self.cfg.TRAINER.COOP.ATK
        results["eps"] = float(self.cfg.AT.EPS)
        for name, value in results["overall"].items():
            self.write_scalar(f"{split}/vq/{name}", value, self.epoch)
        for layer_name, layer in results["layers"].items():
            for name, value in layer["overall"].items():
                self.write_scalar(f"{split}/vq/{layer_name}/{name}", value, self.epoch)
        os.makedirs(self.output_dir, exist_ok=True)
        path = osp.join(self.output_dir, f"quantization_stats_{split}.json")
        sample_path = osp.join(self.output_dir, f"quantization_samples_{split}.jsonl")
        with open(sample_path, "w", encoding="utf-8") as fp:
            for row in pair_stats.samples:
                fp.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        results["samples_file"] = osp.basename(sample_path)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2, allow_nan=False)
        print(f"Quantization pair statistics ({results['num_pairs']} pairs):")
        for name, value in results["overall"].items():
            print(f"  {name}: {value:.6f}")
        for name, group in results["stability_groups"].items():
            print(f"  {name}: n={group['count']}, "
                  f"clean_acc={group['clean_accuracy']}, adv_acc={group['adv_accuracy']}, "
                  f"correct_to_wrong={group['transitions']['correct_to_wrong']}")
        for layer_name, layer in results["layers"].items():
            print(f"  {layer_name}: {layer['overall']}")
        print(f"Saved quantization statistics to {path}")
        print(f"Saved per-sample records to {sample_path}")

    def test(self, split=None, *, collect_vq_stats=False):
        """Run the original evaluation, optionally inspecting its existing attack pairs."""
        if collect_vq_stats and not self.cfg.TRAINER.COOP.ATK_TEST:
            raise ValueError("collect_vq_stats=True requires ATK_TEST=True.")
        pair_stats = ResidualQuantizationPairStats() if collect_vq_stats else None
        self.set_model_mode("eval")
        self.evaluator.reset()

        if self.cfg.TRAINER.COOP.ATK_TEST:
            self.evaluator_adv = build_evaluator(self.cfg, lab2cname=self.lab2cname)
            self.evaluator_adv.reset()
            torch.cuda.empty_cache()
            model = self.model
            eps = self.cfg.AT.EPS
            alpha = eps / 4.0
            steps = 100
            if self.cfg.TRAINER.COOP.ATK == 'aa':
                attack = AutoAttack(model,
                                    norm='Linf',
                                    eps=eps,
                                    version='standard',
                                    verbose=False)
            elif self.cfg.TRAINER.COOP.ATK == 'pgd':
                attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
            elif self.cfg.TRAINER.COOP.ATK == 'tpgd':
                attack = TPGD(model, eps=eps, alpha=alpha, steps=steps)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                output = self.model_inference(input).detach()
            self.evaluator.process(output, label)
            torch.cuda.empty_cache()
            if self.cfg.TRAINER.COOP.ATK_TEST:
                bs = input.size(0)
                model.mode = 'attack'
                if self.cfg.TRAINER.COOP.ATK == 'aa':
                    model = model.float()
                    adv = attack.run_standard_evaluation(input, label, bs=bs)
                elif self.cfg.TRAINER.COOP.ATK in ['pgd', 'tpgd']:
                    adv = attack(input, label)
                else:
                    adv, _ = pgd(input, label, model, CWLoss, eps, alpha, steps)
                model.mode = 'classification'
                with torch.no_grad():
                    output_adv  = self.model_inference(adv).detach()
                self.evaluator_adv.process(output_adv, label)
                if pair_stats is not None:
                    # Reuse this exact pair; never run a second attack for statistics.
                    # Unwrap DataParallel only for this optional diagnostic pass.
                    feature_model = self.model.module if isinstance(
                        self.model, nn.DataParallel
                    ) else self.model
                    clean_features = feature_model.quantization_features(input)
                    adv_features = feature_model.quantization_features(adv.detach())
                    pair_stats.update(
                        clean_features, adv_features, label, output, output_adv,
                        num_codes=[book.num_embeddings for book in feature_model.vqvae.vq_embedding],
                        paths=batch.get("impath"),
                    )
                    del clean_features, adv_features
        if pair_stats is not None:
            self._save_quantization_stats(pair_stats, split)

        results = self.evaluator.evaluate()
        results_adv = {}
        if self.cfg.TRAINER.COOP.ATK_TEST:
            results_adv = self.evaluator_adv.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        if self.cfg.TRAINER.COOP.ATK_TEST:
            for k, v in results_adv.items():
                tag = f"{split}/{k}_adv"
                self.write_scalar(tag, v, self.epoch)
        if self.cfg.TRAINER.COOP.ATK_TEST:
            with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
                json.dump(results, fp) 
                fp.write('\n')
                json.dump(results_adv, fp) 
                fp.write('\n')
            return list(results.values())[0], list(results_adv.values())[0]
        else:
            return list(results.values())[0]

    
def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else torch.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*torch.sign(ig)
    else:
        pert += eps_step*torch.sign(ig)
    pert.clamp_(-eps, eps)
    adv = torch.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert
