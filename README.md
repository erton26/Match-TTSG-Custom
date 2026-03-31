## 環境構築
```
conda create -n match-ttsg-custom python=3.10 -y
conda activate match-ttsg-custom

conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## 学習
1. configs/data 内にデータのyamlファイルを作成（data_statisticsの設定以外）

2. データセットのdata_statisticsを計算し、出力は元のyamlファイルのdata_statisticsに記入
例：
```
python ./match_ttsg/utils/generate_data_statistics_multimodal_deca.py -i multimodal_singlefinetune_abi_verbatim_deca_newfilter_changestats.yaml
```

3. 学習を開始する（config/train.yaml や arg で設定を変更）
モデルファインチューニングの実行例：
```
python ./match_ttsg/train.py experiment=baseline_multimodal_singlefinetune_verbatim data=multimodal_singlefinetune_abi_verbatim_deca_newfilter_changestats model=match_ttsg_custom_deca ckpt_path=/home/git/Match-TTSG/logs/train/multimodal_base_verbatim_baseline/runs/abi_base_decanew/checkpoints/last.ckpt
```

モデルやデータセットの設定を変更したい場合は、config内のyamlファイルで行ってください。

## 推論
推論はこのhuggingfaceのリポジトリを参考してください（学習済みモデルも載っています）：
https://huggingface.co/rton26/match-ttsg-custom
