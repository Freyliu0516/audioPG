import sys
import types
import collections
import collections.abc
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional as F
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple


# Environment patch for compatibility
try:
    import torch._six
except ImportError:
    _six = types.ModuleType('torch._six')
    _six.container_abcs = collections.abc
    _six.string_classes = str
    sys.modules['torch._six'] = _six
    torch._six = _six


class PatchEmbed_new(nn.Module):
    """Patch embedding layer for audio spectrograms."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans  # Store in_chans for get_output_shape
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size, in_chans)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size, in_chans):
        return self.proj(torch.randn(1, in_chans, img_size[0], img_size[1])).shape

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Get 2D sine-cosine positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """Get flexible 2D sine-cosine positional embedding."""
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sine-cosine positional embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sine-cosine positional embedding from grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone for audio."""
    
    def __init__(self, img_size=(1024, 128), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 audio_exp=False, decoder_mode=0, 
                 use_custom_patch=False, pos_trainable=False, **kwargs):
        super().__init__()

        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Encoder
        if use_custom_patch:
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                              embed_dim=embed_dim, stride=stride)
        else:
            # Using stride equal to patch_size for no overlap
            self.patch_embed = PatchEmbed_new(img_size, patch_size, in_chans, embed_dim, stride=patch_size)

        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)
        
        # Encoder transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=pos_trainable)
        self.decoder_mode = decoder_mode
        self.norm_pix_loss = norm_pix_loss
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with sine-cosine embedding."""
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
            
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialization for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert spectrograms to patches."""
        p = self.patch_embed.patch_size[0]
        # Handling for audio (1024, 128)
        if self.audio_exp:
            # Regular grid patchify for audio spectrograms
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x):
        """Convert patches back to spectrograms."""
        p = self.patch_embed.patch_size[0]
        h = 1024 // p  # Fixed height based on audio specs
        w = 128 // p   # Fixed width based on audio specs
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Perform random masking of patches."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Exclude CLS token from pos embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # Add CLS token pos embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]  # Remove CLS token predictions
        return pred

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """Compute reconstruction loss."""
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """Forward pass of the full model."""
        emb_enc, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(emb_enc, ids_restore)
        loss = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    """Create AudioMAE base model with patch size 16."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class AudioMAEClassifier(nn.Module):
    """AudioMAE-based classifier for downstream tasks."""
    
    def __init__(self, img_size=(1024, 128), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 num_classes=50, drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        # Using stride 16 to match pre-trained weights
        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, 
                                          in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, drop_path=dpr[i]) 
            for i in range(depth)])
            
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with sine-cosine embedding."""
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], 
                                                     self.patch_embed.patch_hw, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract features from the model."""
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Global average pooling on spatial tokens (excluding CLS)
        return x[:, 1:, :].mean(dim=1) 

    def forward(self, x):
        """Forward pass through the classifier."""
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained_mae(self, checkpoint_path):
        """Load pre-trained MAE weights for fine-tuning."""
        print(f"📥 Loading pretrained weights from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            # Compatible with both our pretrain.py format and original model format
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            new_dict = {}
            for k, v in state_dict.items():
                if 'decoder' in k or 'mask_token' in k: continue
                new_dict[k] = v
            msg = self.load_state_dict(new_dict, strict=False)
            print(f"✅ Loaded: {msg}")
        except Exception as e:
            print(f"❌ Load failed: {e}")


def get_model_configs():
    """Return common model configurations."""
    configs = {
        'base': {
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
        }
    }
    return configs