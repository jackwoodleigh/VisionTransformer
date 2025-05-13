
import torch
from torch import nn
import yaml
from toolkit.init import init
from toolkit.utils import save_images
from toolkit.transforms import denormalize

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import math # Needed for ceil and sqrt for grid layout

def save_comparison_grid(
    hr_batch: torch.Tensor,
    lr_batch: torch.Tensor,
    output_dir: str,
    filename_prefix: str = "comparison_grid",
    upscale_lr: bool = True,
    resampling_method: Image.Resampling = Image.Resampling.BICUBIC,
    output_format: str = "png" # Allow specifying output format
) -> None:

    # --- Input Validation ---
    if not isinstance(hr_batch, torch.Tensor) or not isinstance(lr_batch, torch.Tensor):
        raise TypeError("Inputs hr_batch and lr_batch must be torch.Tensors.")

    if hr_batch.dim() != 4 or lr_batch.dim() != 4:
        raise AssertionError("Input tensors must have 4 dimensions (B, C, H, W).")

    batch_size = hr_batch.shape[0]
    if batch_size == 0:
        print("Warning: Input batches are empty. No image will be saved.")
        return
        # Alternatively, raise ValueError("Input batch size cannot be zero.")


    if batch_size != lr_batch.shape[0]:
        raise ValueError(f"Batch sizes do not match: "
                         f"HR batch size {hr_batch.shape[0]}, "
                         f"LR batch size {lr_batch.shape[0]}")

    # --- Directory Setup ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Image Processing ---
    to_pil = transforms.ToPILImage()
    comparison_images = []
    single_comp_width = 0
    single_comp_height = 0
    first_image_mode = None

    for i in range(batch_size):
        # Get individual tensors (remove batch dimension)
        hr_tensor = hr_batch[i].detach().cpu()
        lr_tensor = lr_batch[i].detach().cpu()

        # --- Pre-process Tensor Values (Clamp to [0, 1]) ---
        # Adjust this if your tensors have a different range (e.g., [-1, 1])
        hr_tensor = torch.clamp(hr_tensor, 0, 1)
        lr_tensor = torch.clamp(lr_tensor, 0, 1)
        # Example for [-1, 1] range:
        # hr_tensor = (hr_tensor + 1) / 2
        # lr_tensor = (lr_tensor + 1) / 2

        # --- Convert Tensors to PIL Images ---
        try:
            hr_img = to_pil(hr_tensor)
            lr_img = to_pil(lr_tensor)
            if i == 0: # Capture mode from the first successful conversion
                first_image_mode = hr_img.mode
        except Exception as e:
            print(f"Error converting tensor to PIL image for index {i}: {e}")
            print(f"HR tensor shape: {hr_tensor.shape}, dtype: {hr_tensor.dtype}, "
                  f"min: {hr_tensor.min()}, max: {hr_tensor.max()}")
            print(f"LR tensor shape: {lr_tensor.shape}, dtype: {lr_tensor.dtype}, "
                  f"min: {lr_tensor.min()}, max: {lr_tensor.max()}")
            continue # Skip this image pair

        # --- Handle Resolution Difference (Optional Upscaling) ---
        hr_w, hr_h = hr_img.size
        lr_w_orig, lr_h_orig = lr_img.size

        if upscale_lr and (lr_w_orig != hr_w or lr_h_orig != hr_h):
            # print(f"Upscaling LR image {i} from {lr_img.size} to {hr_img.size}")
            lr_img_display = lr_img.resize((hr_w, hr_h), resample=resampling_method)
        else:
            lr_img_display = lr_img # Use original LR image

        lr_w_display, lr_h_display = lr_img_display.size

        # Ensure heights match for side-by-side concatenation
        if hr_h != lr_h_display:
           print(f"Warning: Heights differ for index {i} (HR: {hr_h}, LR: {lr_h_display}). "
                 f"Resizing LR display height to match HR height.")
           aspect_ratio = lr_w_display / lr_h_display
           new_lr_w = int(hr_h * aspect_ratio)
           lr_img_display = lr_img_display.resize((new_lr_w, hr_h), resample=resampling_method)
           lr_w_display, lr_h_display = lr_img_display.size

        # --- Create Side-by-Side Comparison Image ---
        total_width = hr_w + lr_w_display
        combined_height = hr_h # Heights must now match

        # Ensure consistent image mode (e.g., 'RGB') using the mode of the first HR image
        if hr_img.mode != first_image_mode:
             print(f"Warning: HR image mode ({hr_img.mode}) differs from first image mode ({first_image_mode}) for index {i}. "
                   f"Converting HR to {first_image_mode}.")
             hr_img = hr_img.convert(first_image_mode)
        if lr_img_display.mode != first_image_mode:
             print(f"Warning: LR display image mode ({lr_img_display.mode}) differs from first image mode ({first_image_mode}) for index {i}. "
                   f"Converting LR display to {first_image_mode}.")
             lr_img_display = lr_img_display.convert(first_image_mode)

        combined_img = Image.new(first_image_mode, (total_width, combined_height))
        combined_img.paste(hr_img, (0, 0))
        combined_img.paste(lr_img_display, (hr_w, 0))

        # Store the combined image and its dimensions (only need dimensions once)
        comparison_images.append(combined_img)
        if i == 0:
            single_comp_width = total_width
            single_comp_height = combined_height

    # --- Grid Creation (if any images were processed) ---
    if not comparison_images:
        print("No valid comparison images were generated.")
        return

    if not first_image_mode:
         print("Error: Could not determine image mode. Cannot create grid.")
         return

    num_images = len(comparison_images)

    # Determine grid size (try to make it squarish)
    grid_cols = int(math.ceil(math.sqrt(num_images)))
    grid_rows = int(math.ceil(num_images / grid_cols))

    # Create the grid canvas
    grid_width = grid_cols * single_comp_width
    grid_height = grid_rows * single_comp_height
    grid_image = Image.new(first_image_mode, (grid_width, grid_height))

    # Paste images into the grid
    for idx, img in enumerate(comparison_images):
        row = idx // grid_cols
        col = idx % grid_cols
        paste_x = col * single_comp_width
        paste_y = row * single_comp_height
        grid_image.paste(img, (paste_x, paste_y))

    # --- Save the Final Grid Image ---
    filename = f"{filename_prefix}.{output_format.lower()}"
    output_path = os.path.join(output_dir, filename)

    try:
        grid_image.save(output_path)
        print(f"Saved comparison grid image to: {output_path}")
    except Exception as e:
        print(f"Error saving grid image {output_path}: {e}")

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = init(config, 0, 0)

hr_b = []
lr_b = []
for i in range(100):
    hr, lr = train_dataset[i]
    hr_b.append(denormalize(hr))
    lr_b.append(denormalize(lr))

save_comparison_grid(
        hr_batch=torch.stack(hr_b, dim=0),
        lr_batch=torch.stack(lr_b, dim=0),
        output_dir="./comparison_images",
        filename_prefix="test_comp",
        upscale_lr=False, # Try True and False
        resampling_method=Image.Resampling.NEAREST # Try NEAREST or BICUBIC
    )
