import math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------


def patchify(x, patch_size):
    """x: (B, C, H, W) -> (B, num_patches, patch_dim)"""
    B, C, H, W = x.shape
    p = patch_size
    if H % p != 0 or W % p != 0: return # terminate if height and width are not divisible by patch size
    x = x.reshape(B, C, H // p, p, W // p, p) #patchify
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C) #patches are laid out in sequence and flattened
    return x

def unpatchify(x, patch_size, img_channels, H, W):
    """x: (B, num_patches, p*p*C) -> (B, C, H, W)"""
    B, N, PPc = x.shape 
    p = patch_size
    h, w = H // p, W // p
    x = x.reshape(B, h, w, p, p, img_channels)  # restitch patch
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, img_channels, H, W)  # merge back to full image
    return x

def sobel_edges(batch_rgb):
    """Compute simple Sobel edge magnitude (grayscale) for a batch of RGB images in [0,1]."""
    B, C, H, W = batch_rgb.shape
    if C == 3: #for RGB
        r, g, b = batch_rgb[:,0:1], batch_rgb[:,1:2], batch_rgb[:,2:3] 
        gray = 0.299*r + 0.587*g + 0.114*b  # (B,1,H,W) # convert to grayscale with standard rgb weights
    elif C == 1:
        gray = batch_rgb  # already grayscale
    else:
        raise ValueError(f"sobel_edges expects 1 or 3 channels, got {C}")

    # Sobel kernels
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1) #convolute
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy) # com,pute gradient magnitue
    mag = (mag - mag.amin(dim=(2,3), keepdim=True)) / (mag.amax(dim=(2,3), keepdim=True) - mag.amin(dim=(2,3), keepdim=True) + 1e-6)
    #nortmalize to [0,1]
    return mag  # (B,1,H,W)

# -----------------------------
# Transformer building blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*hidden_mult)
        self.fc2 = nn.Linear(dim*hidden_mult, dim)
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult=4, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_mult, p)
    def forward(self, x):
        # Self-attention
        a = self.ln1(x)
        y, _ = self.attn(a, a, a, need_weights=False)
        x = x + y
        # MLP
        y = self.mlp(self.ln2(x))
        x = x + y
        return x

# class definitiion of multimodal VIT
class MultiModalViT(nn.Module):
    def __init__(self, img_channels, edge_channels, img_size, patch_size, dim, depth, heads, mlp_mult, mask_ratio):
        super().__init__() #inherit superclass stuff
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        n_patches = (img_size // patch_size) ** 2
        patch_dim_img  = (patch_size**2) * img_channels # number of pixels per image patch
        patch_dim_edge = (patch_size**2) * edge_channels # number of pixels per edge patch

        
        self.img_embed  = nn.Linear(patch_dim_img,  dim) # linear layers for embedding tokens
        self.edge_embed = nn.Linear(patch_dim_edge, dim)

        # type embeddings to tell the transformer which tokens are which
        self.modality_embed = nn.Embedding(2, dim)  # 0=RGB, 1=EDGE 

        # positional embeddings (shared for all tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches*2, dim) * 0.02)  # we concatenate two streams?

        # initialize encoder layers
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_mult) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim) # establish layernorm

        # pixel values to reconstruct rgb image
        self.to_patch_rgb = nn.Linear(dim, patch_dim_img)

        # only compute loss on masked patches!!!
        self.n_patches = n_patches

    # feed forward method
    def forward(self, rgb, edge):
        """
        rgb:  (B,3,H,W) in [0,1]
        edge: (B,1,H,W) in [0,1]
        """
        B, _, H, W = rgb.shape # batch size, channels, height, width
        p = self.patch_size

        # patchify both modalities
        rgb_patches  = patchify(rgb,  p)  # (batch, number of patches, domensions
        edge_patches = patchify(edge, p)  

        # embed the patches + add modality IDS (self.modality_embed for rewcognizing embeddings)
        rgb_tok  = self.img_embed(rgb_patches)  + self.modality_embed(torch.zeros((B, self.n_patches), dtype=torch.long, device=rgb.device))
        edge_tok = self.edge_embed(edge_patches) + self.modality_embed(torch.ones ((B, self.n_patches), dtype=torch.long, device=rgb.device))

        # concatenate both modaklities' srquences, along with adding positinal embeddings
        tokens = torch.cat([rgb_tok, edge_tok], dim=1)  # (B, 2N, D)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # transformer encoder
        for blk in self.blocks:
            tokens = blk(tokens) # each block contains self attention + MLP
        tokens = self.norm(tokens) #layernorm 

        # split the streams back (keep alignment by patch index)
        rgb_tokens = tokens[:, :self.n_patches, :]   # (batch, patches, dimensions

        # reconstruction of RGB patches from RGB tokens (fused with EDGE via the slef-attention)
        rgb_patch_pred = self.to_patch_rgb(rgb_tokens)  

        return rgb_patch_pred 

    def random_mask(self, x):
        """
        x: (B, N, D) -> returns mask of shape (B, N) with True for masked positions
        """
        B, N, _ = x.shape
        n_mask = int(self.mask_ratio * N)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            mask[b, perm[:n_mask]] = True
        return mask


def get_loaders(batch_size=512, img_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # macOS/MPS tends to be happiest with 0 workers and no pin_memory
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


# FOR MULTI IMAGE
# def get_loaders(batch_size=512):
#     transform = transforms.Compose([
#         transforms.Resize((84, 84)),
#         # transforms.Grayscale(num_output_channels=1), #for single channel images
#         transforms.ToTensor()
#     ])

#     dataset = ImageFolder("./data/metaworld_frames/pick-place-v3", transform=transform)
#     n = len(dataset)
#     n_train = int(0.9 * n)
#     n_test = n - n_train

#     train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


# FOR SINGLE IMAGE
# def get_loaders(batch_size=512):
#     transform = transforms.Compose([
#         transforms.Resize((84, 84)),
#         # transforms.Grayscale(num_output_channels=1), # for single channel images
#         transforms.ToTensor()
#     ])

#     dataset = ImageFolder("./data/metaworld_frames/pick-place-v3", transform=transform)

#     # --- Single-image mode ---
#     # Instead of splitting into train/test and using large batches,
#     # just take the first image and build tiny loaders around it.
#     from torch.utils.data import Subset
#     if len(dataset) == 0:
#         raise RuntimeError("No images found in ./data/metaworld_frames/pick-place-v3")
#     one_ds = Subset(dataset, [0])  # pick a single sample (index 0)

#     train_loader = DataLoader(one_ds, batch_size=1, shuffle=False)
#     test_loader  = DataLoader(one_ds, batch_size=1, shuffle=False)
#     return train_loader, test_loader


def train_one_epoch(model, opt, loader, device, patch_size):
    model.train() # set training to true
    total = 0.0 # total sum of losses
    for imgs, _ in loader: #iteratve the batches
        imgs = imgs.to(device)  # (B,3,32,32)
        edges = sobel_edges(imgs)  # (B,1,32,32)

        # Forward
        pred_patches = model(imgs, edges)  # (B,N,PP*3) #compute sobel edge map

        # Masking for MAE-style loss
        with torch.no_grad(): #do not track gradients
            gt_patches = patchify(imgs, patch_size) #patchify original images
            # create mask based on tokens produced (same N)
            dummy_tokens = pred_patches.new_zeros(pred_patches.shape)  # just for shape
        mask = model.random_mask(pred_patches)  # (B,N) #mask patches

        # Compute loss ONLY on masked patches
        loss = ((pred_patches[mask] - gt_patches[mask])**2).mean() # MSE 


        # backpropagate and update weights, clkearing old gradiebnts
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item() * imgs.size(0) #accumulate the sum of loss over samplse of the batch
    return total / len(loader.dataset) # 1/n

@torch.no_grad() #not neded
def evaluate_and_visualize(model, loader, device, patch_size, out_path="recon_grid.png"):
    model.eval() # set training to false
    imgs, _ = next(iter(loader)) #grab first batch of images from the loader
    imgs = imgs.to(device)[:16] #move images to device and keep the first 16 samples per se
    edges = sobel_edges(imgs) # compute the sobel edge maps
    B, _, H, W = imgs.shape # records batch size, height and width

    pred_patches = model(imgs, edges)  # forward pass
    recons = unpatchify(pred_patches, patch_size, img_channels=3, H=H, W=W).clamp(0,1) #unpatchify

    # stack the originals above the reconstructed and display both
    grid = utils.make_grid(torch.cat([imgs.cpu(), recons.cpu()], dim=0), nrow=16, padding=2)
    plt.figure(figsize=(16,4))
    plt.axis("off")
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"saved reconstruction grid to {out_path}")

    # to show differences, betweene original and reconsturcted
    # # Absolute per-pixel difference, amplified for visibility
    # diff = (recons - imgs).abs().clamp(0, 1)
    # diff_amp = (diff * 5.0).clamp(0, 1)  # amplify subtle errors

    # # Save a grid of amplified differences aligned with originals
    # diff_grid = utils.make_grid(diff_amp.cpu(), nrow=16, padding=2)
    # plt.figure(figsize=(16, 2))
    # plt.axis("off")
    # plt.imshow(diff_grid.permute(1,2,0).numpy())
    # diff_path = out_path.replace(".png", "_diff.png")
    # plt.tight_layout()
    # plt.savefig(diff_path, dpi=160)
    # print(f"Saved amplified difference grid to {diff_path}")

    # Report simple metrics on this batch
    mse  = F.mse_loss(recons, imgs).item()
    psnr = (20 * torch.log10(torch.tensor(1.0, device=device))
            - 10 * torch.log10(F.mse_loss(recons, imgs) + 1e-12)).item()
    print(f"Compare metrics — MSE: {mse:.6f},  PSNR: {psnr:.2f} dB")

@torch.no_grad()
def evaluate_single_image(model, loader, device, patch_size, out_path="./out/recon_single.png", idx=0):
    model.eval()
    imgs, _ = next(iter(loader))
    img = imgs[idx].to(device)
    edges = sobel_edges(img.unsqueeze(0))
    B, C, H, W = img.unsqueeze(0).shape

    pred_patches = model(img.unsqueeze(0), edges)  # (1,N,PP*3)
    recon = unpatchify(pred_patches, patch_size, img_channels=3, H=H, W=W).clamp(0,1)

    # Side-by-side original and reconstruction
    grid = utils.make_grid(torch.cat([img.cpu().unsqueeze(0), recon[0].cpu().unsqueeze(0)], dim=0), nrow=2, padding=2)
    utils.save_image(grid, out_path)
    print(f"Saved single image reconstruction to {out_path}")

    mse = F.mse_loss(recon, img.unsqueeze(0)).item()
    psnr = (20 * torch.log10(torch.tensor(1.0, device=device))
            - 10 * torch.log10(F.mse_loss(recon, img.unsqueeze(0)) + 1e-12)).item()
    print(f"Single image metrics — MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

@torch.no_grad()
def visualize_mask(model, img, patch_size, out_path="./out/masked_image.png"):
    """
    Show which patches are masked (grayed out) for visualization.
    """
    model.eval()
    B, C, H, W = img.shape
    edge = sobel_edges(img)

    # Forward once to create a dummy prediction (only to get N)
    pred_patches = model(img, edge)
    mask = model.random_mask(pred_patches)  # (B, N)

    # Unpatchify to build a binary mask image
    p = patch_size
    N = (H // p) * (W // p)
    mask_img = torch.zeros_like(img)
    idx = 0
    for i in range(0, H, p):
        for j in range(0, W, p):
            if mask[0, idx]:
                mask_img[:, :, i:i+p, j:j+p] = 1.0  # mark masked patches white
            idx += 1

    # Overlay: dim masked patches
    masked_vis = img * (1 - mask_img) + mask_img * 0.8
    utils.save_image(masked_vis, out_path)
    print(f"Saved masked image visualization to {out_path}")


# -----------------------------
# Batch Mask Visualization Helper
# -----------------------------
@torch.no_grad()
def visualize_mask_batch(model, loader, device, patch_size, out_path="./out/masked_batch.png"):
    """
    Visualize a batch of images with masked patches (dimmed), saving a grid.
    """
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    edges = sobel_edges(imgs)
    B, C, H, W = imgs.shape

    # Forward once to create a dummy prediction (only to get N)
    pred_patches = model(imgs, edges)
    mask = model.random_mask(pred_patches)  # (B, N)

    # Build binary mask images for each sample in the batch
    p = patch_size
    N = (H // p) * (W // p)
    mask_imgs = torch.zeros_like(imgs)
    for b in range(B):
        idx = 0
        for i in range(0, H, p):
            for j in range(0, W, p):
                if mask[b, idx]:
                    mask_imgs[b, :, i:i+p, j:j+p] = 1.0
                idx += 1
    

    # Overlay: dim masked patches for each image in the batch
    masked_vis = imgs * (1 - mask_imgs) + mask_imgs * 0.8

    # Create a grid of masked images (up to 16 per row)
    grid = utils.make_grid(masked_vis.cpu(), nrow=16, padding=2)
    utils.save_image(grid, out_path)
    print(f"Saved masked batch visualization to {out_path}")

if __name__ == "__main__":
    
    # fix the randomness just for comparison across multiple runs
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    img_size   = 84 # for grayscale images, use 28

    # HYPERPARAMS
    #do not change this, 8 is too coarse:
    patch_size = 7  # size for each square patch, so in this case 84/7 = 12 x 12 patches
    dim = 128 # embedding
    depth = 4 # encoders in transformer
    heads = 4 #for multi-head attention
    batch_size = 16 # if im doing the 16 images per batch, i can do 512
    lr = 3e-4 
    epochs = 3000 #75 # 1500 if dealing with single image

    train_loader, test_loader = get_loaders(batch_size, img_size) #load training and testing data (9:1)
    
    device = torch.device("mps") # use GPU

    #for grayscale images, use img_channels=1
    model = MultiModalViT(img_channels=1, edge_channels=1, img_size=img_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_mult=4, mask_ratio=0.75).to(device) # run on GPU

    visualize_mask_batch(model, test_loader, device, patch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05) #to update the weights
    # visualize_mask(model, img_for_single_eval.unsqueeze(0), patch_size, out_path="./out/masked_image.png")

    for ep in range(1, epochs+1): #over x epochs
        train_loss = train_one_epoch(model, opt, train_loader, device, patch_size) #get training loss (MSE)
        print(f"Epoch {ep:02d} | train MSE(masked) = {train_loss:.4f}") # output the lostt
        if ep % 2 == 0: #for every other epoch output result
            imgs_for_single_eval, _ = next(iter(test_loader))
            img_for_single_eval = imgs_for_single_eval[0].to(device)
            evaluate_single_image(model, test_loader, device, patch_size, out_path="./out/recon_single.png", idx=0)
            evaluate_and_visualize(model, test_loader, device, patch_size, out_path=f"./out/recon_grid_ep{ep}.png")

    evaluate_and_visualize(model, test_loader, device, patch_size, out_path="recon_final.png")
    
    


