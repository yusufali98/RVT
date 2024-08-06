import torch
import torch.nn.functional as F

def concatenate_tensors_with_padding(demo_imgs, virtual_images):
    # Find the maximum length of sequences in demo_imgs
    max_length = max(demo_tensor.shape[0] for demo_tensor in demo_imgs)

    # Concatenate with padding and create masks
    concatenated_imgs = []
    masks = []
    for i in range(len(demo_imgs)):
        demo_tensor = demo_imgs[i]
        virtual_tensor = virtual_images[i]

        # Create mask for demo_tensor
        demo_mask = torch.ones(demo_tensor.shape[0], 1, 1)

        # Pad the demo_tensor and mask to match the max_length
        padded_demo_tensor = F.pad(demo_tensor, (0, 0, 0, max_length - demo_tensor.shape[0]))
        padded_demo_mask = F.pad(demo_mask, (0, 0, 0, max_length - demo_mask.shape[0]))

        # Adjust dimensions of virtual_tensor to match demo_tensor
        virtual_tensor = virtual_tensor.unsqueeze(1).expand(-1, 64, 128)

        # Concatenate along the height dimension (dim=0)
        concatenated_tensor = torch.cat((padded_demo_tensor, virtual_tensor), dim=0)
        concatenated_imgs.append(concatenated_tensor)

        # Create mask for virtual_tensor (no padding for virtual_tensor)
        virtual_mask = torch.ones(virtual_tensor.shape[0], 1, 1)

        # Concatenate masks along the height dimension (dim=0)
        concatenated_mask = torch.cat((padded_demo_mask, virtual_mask), dim=0)
        masks.append(concatenated_mask)

    # Convert the list of tensors back to a single tensor if needed
    result = torch.stack(concatenated_imgs)
    mask_result = torch.stack(masks)

    return result, mask_result, max_length

def extract_tensors(concatenated_tensor, mask, max_length):
    demo_tensors = []
    virtual_tensors = []
    for i in range(concatenated_tensor.shape[0]):
        # Determine the original length of the demo_tensor using the mask
        demo_length = (mask[i, :max_length, 0, 0] == 1).nonzero(as_tuple=True)[0].shape[0]
        demo_tensor = concatenated_tensor[i, :demo_length, :, :]
        virtual_tensor = concatenated_tensor[i, demo_length:, :, :]
        virtual_tensor = virtual_tensor.view(-1, 64, 128)
        demo_tensors.append(demo_tensor)
        virtual_tensors.append(virtual_tensor)
    
    return demo_tensors, virtual_tensors

# Example initialization
# Replace with actual tensors
demo_imgs = [torch.randn((n, 64, 128)) for n in range(1, 25)]
virtual_images = torch.randn((24, 768, 128))

# Concatenate tensors with padding and create masks
result, mask_result, max_length = concatenate_tensors_with_padding(demo_imgs, virtual_images)

# Re-extract the demo_tensor and virtual_image from the sequence
re_extracted_demo_tensors, re_extracted_virtual_tensors = extract_tensors(result, mask_result, max_length)

# Unit Test
def test_extraction(demo_imgs, virtual_images, re_extracted_demo_tensors, re_extracted_virtual_tensors):
    for original_demo, re_extracted_demo, re_extracted_virtual in zip(demo_imgs, re_extracted_demo_tensors, re_extracted_virtual_tensors):
        assert torch.allclose(original_demo, re_extracted_demo, atol=1e-6), f"Demo tensor does not match: {original_demo.shape} vs {re_extracted_demo.shape}"
        assert re_extracted_virtual.shape == (768, 64, 128), f"Virtual tensor shape mismatch: expected (768, 64, 128), got {re_extracted_virtual.shape}"
    print("All tests passed.")

# Run the unit test
test_extraction(demo_imgs, virtual_images, re_extracted_demo_tensors, re_extracted_virtual_tensors)
