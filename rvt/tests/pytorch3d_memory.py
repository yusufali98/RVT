import torch
from rvt.mvt.renderer import get_cube_R_T
from pytorch3d.renderer import FoVOrthographicCameras

h = 440
max_size = 330

R, T, scale = get_cube_R_T(with_scale=True)
fix_cam = FoVOrthographicCameras(device='cuda',R=R, T=T, znear=0.01, scale_xyz=scale)

pts = torch.randn((1,h*h*h,3)).to('cuda')

pt_scr = fix_cam.transform_points_screen(pts.view(-1,3), image_size=(h,h))

def process_points_in_batches(pts, batch_size, process_fn):
    num_pts = pts.shape[1]
    processed_pts = []
    for i in range(0, num_pts, batch_size):
        print("current range: ", i, i + batch_size)
        batch_pts = pts[:,i:i+batch_size]
        processed_batch = process_fn(batch_pts)
        processed_pts.append(processed_batch)
    return torch.cat(processed_pts, dim=1)

batch_size = max_size * max_size * max_size  # Adjust as needed
processed_pts = process_points_in_batches(pts, batch_size, lambda x: fix_cam.transform_points_screen(x.view(-1, 3), image_size=(h, h)))

def approx_equal(tensor1, tensor2, threshold):
    # Compute the absolute difference between the tensors
    diff = torch.abs(tensor1 - tensor2)
    
    # Check if all elements are within the threshold
    return torch.all(diff <= threshold)

# Define the threshold
threshold = 0.0002

# Check if tensor1 is approximately equal to tensor2 based on the threshold
print(approx_equal(pt_scr, processed_pts, threshold))  # Output: True
