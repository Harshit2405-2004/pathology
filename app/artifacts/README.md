# App files

Copy these into this folder (details and the Drive folder names are in `guide.md`):

```
artifacts/
  pcam_best.pt            Part 1: the patch model in fp16 (~23 MB)
  nuclei_best.pt          Part 3: the nuclei U-Net in fp16 (~49 MB)
  examples/pcam/          Part 1: 4 tumour + 4 normal test patches
  examples/pannuke/       Part 3: 6 test tiles
  heatmaps/               Part 2: 4 slides, each as .npz + .json + _overlay.png
```
A missing file only switches off its own tab.
