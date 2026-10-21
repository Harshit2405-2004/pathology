# streamlit app for the pathology project
# run with:  streamlit run app/app.py
# the files come from the app_artifacts folders on drive (saved at the end of Part 1, 2 and 3),
# copy their contents into app/artifacts/  (see guide.md)

import os
import glob
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology
from skimage.segmentation import find_boundaries, watershed

artifacts = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
types = ['background', 'neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']
colors = np.array([[0, 0, 0], [255, 0, 0], [0, 200, 0], [0, 90, 255], [255, 220, 0], [255, 120, 0]])
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.set_num_threads(2)          # streamlit cloud gives a small cpu


# model (loaded once and kept in memory by streamlit)

@st.cache_resource
def load_patch_model():
    path = artifacts + '/pcam_best.pt'
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location = 'cpu', weights_only = False)
    model = torchvision.models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint['state_dict'])      # the fp16 weights are copied into the normal fp32 model
    model.eval()
    return model


@st.cache_resource
def load_nuclei_model():
    path = artifacts + '/nuclei_best.pt'
    if not os.path.exists(path):
        return None, 0.4
    import segmentation_models_pytorch as smp
    checkpoint = torch.load(path, map_location = 'cpu', weights_only = False)
    model = smp.Unet(encoder_name = checkpoint['config']['encoder'], encoder_weights = None, in_channels = 3, classes = 2 + len(types))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['config'].get('marker_thr', 0.4)


def to_input(image):
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255
    return (x - mean) / std


def predict_patch(model, image, use_tta):
    # with tta: average over the 4 rotations, each with and without a flip
    x = to_input(image)
    probs = []
    with torch.no_grad():
        for k in range(4):
            rotated = torch.rot90(x, k, dims = (2, 3))
            probs.append(torch.sigmoid(model(rotated)).item())
            if use_tta == False:
                break
            probs.append(torch.sigmoid(model(torch.flip(rotated, dims = (3,)))).item())
    return float(np.mean(probs))


# watershed, the same as in part 3

def remove_small(mask, size):
    try:
        return morphology.remove_small_objects(mask, max_size = size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size = size + 1)


def split_nuclei(nucleus_prob, dist, marker_thr = 0.4):
    mask = remove_small(nucleus_prob > 0.5, 10)
    dist = ndi.gaussian_filter(dist.astype(np.float32), sigma = 1)
    markers, _ = ndi.label((dist > marker_thr) & mask)
    markers = remove_small(markers, 3)
    inst = watershed(-dist, markers, mask = mask)
    leftovers, n = ndi.label(mask & (inst == 0))
    if n > 0:
        inst = np.where(leftovers > 0, leftovers + inst.max(), inst)
    inst = remove_small(inst, 10)
    ids = np.unique(inst[inst > 0])
    new_ids = np.zeros(inst.max() + 1, dtype = np.int32)
    new_ids[ids] = np.arange(1, len(ids) + 1)
    return new_ids[inst]


def nucleus_types(inst, type_probs):
    pixel_type = type_probs[1:].argmax(axis = 0) + 1
    result = {}
    for k, box in enumerate(ndi.find_objects(inst), start = 1):
        if box is not None:
            result[k] = int(np.bincount(pixel_type[box][inst[box] == k]).argmax())
    return result


def choose_image(key, folder, size):
    uploaded = st.file_uploader('Upload an H&E tile (png / jpg)', type = ['png', 'jpg', 'jpeg'], key = key)
    examples = sorted(glob.glob(f'{artifacts}/examples/{folder}/*.png'))
    choice = '-'
    if len(examples) > 0:
        choice = st.selectbox('...or pick an example', ['-'] + [os.path.basename(e) for e in examples], key = key + '_example')

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
    elif choice != '-':
        image = Image.open(f'{artifacts}/examples/{folder}/{choice}').convert('RGB')
    else:
        return None

    if image.size != (size, size):
        # cut the middle square and resize it, the magnification still has to be about right
        w, h = image.size
        s = min(w, h)
        image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))
        st.info(f'Resized to {size} x {size} px. The result only makes sense at the magnification the model was trained on.')
    return np.array(image)


def slide_score(heatmap):
    # the main slide score from Part 2: max of the heatmap after a 3 x 3 average
    p = np.nan_to_num(heatmap, nan = 0.0)
    return float(ndi.uniform_filter(p, size = 3, mode = 'constant').max())


@st.cache_data
def load_slides():
    rows = []
    for f in sorted(glob.glob(artifacts + '/heatmaps/*.npz')):
        name = os.path.basename(f).replace('.npz', '')
        heatmap = np.load(f)['prob'].astype(np.float32)
        with open(f'{artifacts}/heatmaps/{name}.json') as fh:
            info = json.load(fh)
        cell_mm2 = (info['cell_um'] / 1000) ** 2
        rows.append({'slide': name,
                     'slide score': round(slide_score(heatmap), 3),
                     'tumour area at 0.5 (mm2)': round(float((np.nan_to_num(heatmap) >= 0.5).sum() * cell_mm2), 2),
                     'tissue (mm2)': round(float((~np.isnan(heatmap)).sum() * cell_mm2), 1),
                     'windows': info['windows'],
                     'seconds': round(info['seconds_total']),
                     'windows per second': round(info['windows_per_second'])})
    if len(rows) == 0:
        return None
    return pd.DataFrame(rows).set_index('slide')


# page

st.set_page_config(page_title = 'Metastasis detection', layout = 'wide')
st.title('Lymph node metastasis detection and nucleus counting')
st.caption('Student project on public data (PatchCamelyon, CAMELYON16, PanNuke). Not a medical device.')

with st.sidebar:
    st.header('Results')
    st.markdown('''
- Patch AUC (PCam test): **0.941**, 0.949 with TTA
- Slide AUC (40 CAMELYON16 test slides): **0.945**
- Hospital 0 vs 1: AUC 1.00 vs 0.82
- Nuclei (PanNuke test fold): Dice **0.80**, bPQ **0.66**, count error **3.1** per tile
- Speed: ~8,400 windows/s on a T4 GPU
''')
    st.caption('Numbers from the notebooks, see RESULTS.md in the repo.')

tab1, tab2, tab3 = st.tabs(['1. Tumour patch', '2. Whole slide heatmaps', '3. Nuclei'])

with tab1:
    st.write('A 96 x 96 px patch at **10x**. The answer is about the centre 32 x 32 px, like in PatchCamelyon.')
    patch_model = load_patch_model()
    if patch_model is None:
        st.warning('app/artifacts/pcam_best.pt not found. See guide.md.')
    else:
        image = choose_image('patch', 'pcam', 96)
        use_tta = st.checkbox('Average over the 8 rotations / flips (TTA)', value = True)
        if image is not None:
            prob = predict_patch(patch_model, image, use_tta)
            col1, col2 = st.columns([1, 2])
            shown = image.copy()
            shown[32, 32:64] = shown[63, 32:64] = shown[32:64, 32] = shown[32:64, 63] = [255, 0, 0]     # the centre box
            col1.image(Image.fromarray(shown).resize((288, 288), Image.NEAREST), caption = 'red box = the region the label is about')
            col2.metric('Tumour probability', f'{prob:.3f}')
            if prob >= 0.5:
                col2.write('**Tumour in the centre region**')
            else:
                col2.write('**No tumour in the centre region**')
            col2.caption('Cut-off 0.5. In the notebook a cut-off of 0.054 was needed to catch 90% of tumour patches '
                         'on validation, because the model is over-confident on slides it has not seen.')

with tab2:
    slides = load_slides()
    if slides is None:
        st.warning('No heatmaps in app/artifacts/heatmaps. See guide.md.')
    else:
        st.write('Precomputed in Part 2 (reading a 2 GB slide needs a GPU and OpenSlide). '
                 'The slide score is the max of the heatmap after a 3 x 3 average.')
        name = st.selectbox('Slide', slides.index.tolist())
        col1, col2 = st.columns([3, 1])
        overlay_path = f'{artifacts}/heatmaps/{name}_overlay.png'
        if os.path.exists(overlay_path):
            col1.image(overlay_path, caption = 'Tumour probability (yellow to red)')
        else:
            col1.warning(f'{name}_overlay.png not found')
        row = slides.loc[name]
        col2.metric('Slide score', f"{row['slide score']:.3f}")
        col2.metric('Tissue area', f"{row['tissue (mm2)']:.1f} mm2")
        col2.metric('Predicted tumour area (cells >= 0.5)', f"{row['tumour area at 0.5 (mm2)']:.2f} mm2")
        col2.caption(f"{row['windows']:,} windows in {row['seconds']} s ({row['windows per second']:,} per second)")

        st.subheader('All slides in the app')
        st.dataframe(slides)
        st.caption('Compare slide scores with each other, not with 0.5: in the test set every normal slide scored at least 0.70. '
                   'The ranking is what works (AUC 0.945); a real cut-off has to be chosen on held-out slides.')

with tab3:
    st.write('A 256 x 256 px tile at **40x** (0.25 microns per pixel), like in PanNuke. A U-Net gives the nucleus mask, '
             'a distance map and the type of every pixel, then watershed splits touching nuclei.')
    nuclei_model, marker_thr = load_nuclei_model()
    if nuclei_model is None:
        st.warning('app/artifacts/nuclei_best.pt not found. See guide.md.')
    else:
        image = choose_image('nuclei', 'pannuke', 256)
        if image is not None:
            with torch.no_grad():
                out = nuclei_model(to_input(image))[0].float()
            inst = split_nuclei(torch.sigmoid(out[0]).numpy(), torch.sigmoid(out[1]).numpy(), marker_thr)
            nucleus_type = nucleus_types(inst, torch.softmax(out[2:], dim = 0).numpy())

            type_image = np.zeros(inst.shape, dtype = np.uint8)
            for k, t in nucleus_type.items():
                type_image[inst == k] = t
            outlined = image.copy()
            border = find_boundaries(inst, mode = 'inner')
            outlined[border] = colors[type_image[border]]

            col1, col2 = st.columns(2)
            col1.image(image, caption = 'Input', width = 320)
            col2.image(outlined, caption = f'{len(nucleus_type)} nuclei', width = 320)
            counts = pd.Series([types[t] for t in nucleus_type.values()], dtype = 'object').value_counts()
            tile_mm2 = (256 * 0.25 / 1000) ** 2
            st.dataframe(pd.DataFrame({'nuclei': counts, 'per mm2': (counts / tile_mm2).round(0)}))
            st.caption('red neoplastic, green inflammatory, blue connective, yellow dead, orange epithelial. '
                       'On the test fold the model counts within about 3 nuclei per tile (of ~25), but it misses small, '
                       'elongated nuclei and never found a "dead" nucleus.')
