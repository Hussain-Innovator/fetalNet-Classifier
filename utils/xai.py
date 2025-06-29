# utils/xai.py
import torch
import numpy as np
import cv2
from PIL import Image

# Import Captum library and its visualization tools
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    Deconvolution,
    GuidedGradCam
)
from captum.attr import visualization as viz

def explain_prediction(model, input_tensor, original_image, class_idx, method='Saliency'):
    """
    A single function to generate explanations using different Captum methods.
    """
    input_tensor.requires_grad = True
    model.eval()
    
    attribution = None

    # --- Choose the attribution algorithm ---
    if method == 'Saliency':
        explainer = Saliency(model)
        attribution = explainer.attribute(input_tensor, target=class_idx)

    elif method == 'Integrated Gradients':
        explainer = IntegratedGradients(model)
        attribution = explainer.attribute(input_tensor, target=class_idx, n_steps=50, internal_batch_size=2)
        
    elif method == 'Guided Backpropagation':
        explainer = GuidedBackprop(model)
        attribution = explainer.attribute(input_tensor, target=class_idx)
        
    elif method == 'Grad-CAM':
        # For Grad-CAM, we need to specify the target layer
        target_layer = model.conv[7] # The final ReLU layer in your FetalNet
        explainer = GuidedGradCam(model, target_layer)
        # GuidedGradCam combines Grad-CAM with Guided Backprop for a sharper image
        attribution = explainer.attribute(input_tensor, target=class_idx)
        
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")

    # --- Process attribution to a visual heatmap ---
    # The visualization method is different for different attribution types
    if method in ['Saliency', 'Integrated Gradients', 'Grad-CAM', 'Guided Backpropagation']:
        # For gradient-based methods, we use a heatmap overlay
        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = np.mean(heatmap, axis=0) # Average across color channels
        heatmap = np.abs(heatmap) # Use absolute values for saliency
        heatmap = np.maximum(heatmap, 0) # Use only positive contributions
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap) # Normalize

        # Use OpenCV to create the visual overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        original_image_cv = cv2.resize(original_image_cv, (224, 224))

        superimposed_img = cv2.addWeighted(original_image_cv, 0.6, heatmap_colored, 0.4, 0)
        final_image = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
    return final_image








# # utils/xai.py
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from model.fetalnet import FetalNetXAI # Import our XAI model wrapper

# def generate_grad_cam_overlay(model, input_tensor, original_image, class_idx):
#     """
#     Generates the Grad-CAM overlay using a tensor hook to reliably capture gradients.
#     """
#     # 1. Wrap the original model in our new FetalNetXAI class
#     xai_model = FetalNetXAI(model)
#     xai_model.eval()

#     # 2. Perform a forward pass to get the feature maps
#     # We don't need to set requires_grad on the input, as the model parameters have it.
#     output, features = xai_model(input_tensor)

#     # --- FIX: Use a hook to capture the gradient of the intermediate 'features' tensor ---
#     captured_gradients = []
#     def save_gradient(grad):
#         captured_gradients.append(grad)
    
#     features.register_hook(save_gradient)
#     # ---------------------------------------------------------------------------------

#     # 3. Perform the backward pass
#     model.zero_grad()
#     output[0, class_idx].backward()

#     # 4. Get the gradient that was captured by our hook
#     gradients = captured_gradients[0]
    
#     # 5. Pool the gradients and compute the heatmap
#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
#     features = features.squeeze(0)

#     for i in range(len(pooled_gradients)):
#         features[i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(features, dim=0).detach().numpy()
#     heatmap = np.maximum(heatmap, 0)
#     # Check for division by zero
#     if np.max(heatmap) > 0:
#         heatmap /= np.max(heatmap)

#     # 6. Use OpenCV to create the visual overlay
#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
#     original_image_cv = cv2.resize(original_image_cv, (224, 224))

#     superimposed_img = cv2.addWeighted(original_image_cv, 0.6, heatmap_colored, 0.4, 0)
#     superimposed_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
#     return superimposed_pil