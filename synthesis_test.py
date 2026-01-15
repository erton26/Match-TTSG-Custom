from match_ttsg.synthesis import Synthetizer

synthetizer = Synthetizer(checkpoint_path = "logs/train/multimodal_base_verbatim_baseline/runs/2025-09-11_14-47-30/checkpoints/last.ckpt",
                          n_timesteps = 50)

test = synthetizer.synthesize("今日は月曜日です。")
print(test["decoder_outputs"].shape)
print(test["waveform"].shape)

print(test["rtf"], test["rtf_w_vocoder"])