# python 3.7
"""Demo."""

import numpy as np
import torch
import streamlit as st
import SessionState

from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import get_gan_type
from psp_sefa_adapter import PSP_MODEL_NAME


@st.cache_resource(show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


@st.cache_data(show_spinner=False)
def factorize_model(_model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(_model, layer_idx)


def sample(model, gan_type, num=1):
    """Samples latent codes."""
    codes = torch.randn(num, model.z_space_dim).cuda()
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    elif gan_type in ['stylegan2', 'psp_stylegan2']:
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    codes = codes.detach().cpu().numpy()
    return codes


@st.cache_data(show_spinner=False)
def synthesize(_model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = _model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2', 'psp_stylegan2']:
        image = _model.synthesis(to_tensor(code))['image']
    image = postprocess(image)[0]
    return image


def reset_semantic_sliders(model_name, layer_idx, num_semantics):
    """Resets all visible semantic sliders back to zero."""
    for sem_idx in range(num_semantics):
        st.session_state.pop(f'semantic_{model_name}_{layer_idx}_{sem_idx}', None)


def main():
    """Main function (loop for StreamLit)."""
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')
    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        [PSP_MODEL_NAME,
         'stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256',
         'pggan_celebahq1024'])

    model = get_model(model_name)
    if model_name == PSP_MODEL_NAME:
        gan_type = 'psp_stylegan2'
    else:
        gan_type = get_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=10, min_value=0, max_value=None, step=1)
    st.sidebar.button(
        'Reset',
        on_click=reset_semantic_sliders,
        args=(model_name, layer_idx, num_semantics))

    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    if gan_type == 'pggan':
        max_step = 5.0
    elif gan_type == 'stylegan':
        max_step = 2.0
    elif gan_type in ['stylegan2', 'psp_stylegan2']:
        max_step = 15.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        slider_key = f'semantic_{model_name}_{layer_idx}_{sem_idx}'
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step,
            value=0.0,
            key=slider_key)

    image_placeholder = st.empty()
    button_placeholder = st.empty()

    try:
        base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model, gan_type)

    state = SessionState.get(model_name=model_name,
                             code_idx=0,
                             codes=base_codes[0:1])
    if state.model_name != model_name:
        state.model_name = model_name
        state.code_idx = 0
        state.codes = base_codes[0:1]

    if button_placeholder.button('Random', key=0):
        state.code_idx += 1
        if state.code_idx < base_codes.shape[0]:
            state.codes = base_codes[state.code_idx][np.newaxis]
        else:
            state.codes = sample(model, gan_type)

    code = state.codes.copy()
    for sem_idx, step in steps.items():
        if gan_type == 'pggan':
            code += boundaries[sem_idx:sem_idx + 1] * step
        elif gan_type in ['stylegan', 'stylegan2', 'psp_stylegan2']:
            valid_layers = [layer for layer in layers if layer < code.shape[1]]
            code[:, valid_layers, :] += boundaries[sem_idx:sem_idx + 1] * step
    image = synthesize(model, gan_type, code)
    image_placeholder.image(image / 255.0)


if __name__ == '__main__':
    main()
