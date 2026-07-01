import torch
import numpy as np
from core.ir152 import IR_152

def load_ir152(weights_path='core/ir152.pth'):
    model = IR_152((112, 112))
    # load state dict directly
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    # Optional: move to GPU if available and if the pipeline supports it
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def compute_ir152_embedding(model, img_batch):
    """
    img_batch: tf.Tensor or numpy array of shape (B, H, W, 3) normalized to [-1, 1].
    IR_152 expects inputs resized to (112, 112) and normalized.
    Returns embedding as numpy array (B, 512).
    """
    if hasattr(img_batch, 'numpy'):
        img_np = img_batch.numpy()
    else:
        img_np = np.array(img_batch)
    
    if len(img_np.shape) == 3:
        img_np = np.expand_dims(img_np, 0)
        
    # TF images are (B, H, W, 3) -> PyTorch (B, 3, H, W)
    img_pt = torch.tensor(img_np, dtype=torch.float32).permute(0, 3, 1, 2)
    
    device = next(model.parameters()).device
    img_pt = img_pt.to(device)
    
    with torch.no_grad():
        emb = model(img_pt)
        # typically normalize the embedding for face recognition evaluation
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        
    return emb.cpu().numpy()
