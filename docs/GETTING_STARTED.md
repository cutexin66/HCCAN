# Getting Started

## Dataset Preparation

Please place the data (RefCOCO, RefCOCO+, RefCOCOg, ReferItGame) or the soft link of dataset folder under ./ln_data/. We follow dataset structure DMS. To accomplish this, the [download_data.sh](../ln_data/download_data.sh) bash script from DMS can be used.

```
cd ./ln_data
bash download_data.sh --path .
```

Please download data indices from [[Gdrive]](https://drive.google.com/file/d/1DYYNQiiJHbn96IoKZtrgDru_NYa_z6m0/view?usp=share_link), and place them as the ./data folder.

```
rm -r ./data
tar -xvf data.tar
```

## Pretrained Checkpoints Preparation

Please download the pretrained detr from [[Gdrive]](https://drive.google.com/drive/folders/1L7vyVV3uWIdd55QV46slvEBDf7ncCUWu?usp=share_link), and put these checkpoints into ./checkpoints.

Note that when pre-train these checkpoints on MSCOCO, the overlapping images of val/test set of corresponding datasets are excluded.