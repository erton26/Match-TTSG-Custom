# MatchTTSG imports
import torch
from match_ttsg.models.match_ttsg_custom_deca import MatchTTSGCustomDECA
from match_ttsg.text import sequence_to_text, text_to_sequence
from match_ttsg.utils.model import denormalize
from match_ttsg.utils.utils import get_user_data_dir, intersperse
#from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    model = MatchTTSGCustomDECA.load_from_checkpoint(checkpoint_path, map_location=device)
    checkpoint = torch.load(checkpoint_path)
    model.eval()
    return model, checkpoint

#model, checkpoint = load_model("logs/train/multimodal_base_verbatim_baseline/runs/abi_base_decanew/checkpoints/last.ckpt")
#model, checkpoint = load_model("logs/train/multimodal_single_abi_verbatim_baseline/runs/2025-02-02_19-37-05/checkpoints/last.ckpt")
model, checkpoint = load_model("logs/train/multimodal_base_verbatim_baseline/runs/abi_base_decanew/checkpoints/last.ckpt")

#print(model.hparams)
#print(checkpoint['global_step'])
print(sum(p.numel() for p in model.parameters()))