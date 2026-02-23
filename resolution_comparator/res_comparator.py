from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

#loading images

ref_img = Image.open('path\\to\\reference\\image.jpg')
comp_img = Image.open('path\\to\\comparative\\image.jpg')

ref_width, ref_height = ref_img.size
comp_width, comp_height = comp_img.size

target_width = min(ref_width, comp_width)
target_height = min(ref_height, comp_height)

#initialize while loop
pic = 1
initial_ssim = 0
ssim_scores = []
resolutions = []


while target_width > 7 and target_height > 7:
    new_width = int(target_width * 0.9)
    new_height = int(target_height * 0.9)
    
    # Resize both images
    resized_ref = ref_img.resize((new_width, new_height), Image.LANCZOS)
    resized_comp = comp_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert both to grayscale
    ref_gray = resized_ref.convert('L')
    comp_gray = resized_comp.convert('L')
    ref_array = np.array(ref_gray)
    comp_array = np.array(comp_gray)
    
    # Calculate
    score = ssim(ref_array, comp_array)
    ssim_scores.append(score)
    resolutions.append(f"{new_width}x{new_height}")

    #saving max SSIM images
    if score > initial_ssim:
        initial_ssim = score
        resized_ref.save(f'path\\to\\save\\file\\pic1_new.jpg')
        resized_comp.save(f'path\\to\\save\\file\\pic2_new.jpg')
    
    print(f'Resized {pic}: {new_width}x{new_height}, SSIM: {score:.4f}')
    
    # Update dimensions
    target_width = new_width
    target_height = new_height
    pic += 1

max_ssim = max(ssim_scores)
max_index = ssim_scores.index(max_ssim)
max_res = resolutions[max_index]

print(f"\n*** Max SSIM: {max_ssim:.4f} at {max_res} ***")

#plot
plt.figure(figsize=(10, 6))
plt.plot(ssim_scores, marker='o')
plt.axhline(y=max_ssim, color='r', linestyle='--', label=f'Max: {max_ssim:.4f}')
plt.scatter(max_index, max_ssim, color='red', s=100, zorder=5)
plt.xlabel('Iteration')
plt.ylabel('SSIM Score')
plt.title('SSIM vs Resolution Degradation')
plt.legend()
plt.grid(True)
plt.show()