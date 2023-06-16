import json
import logging
import os
import re
import subprocess
import time
import traceback

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
import torch
from scipy.io import wavfile

from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def mix_submit_click(js, mode):
    try:
        assert js.lstrip() != ""
        modes = {"凸组合": 0, "线性组合": 1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path, mix_rate = zip(*data)
        path = mix_model(model_path, mix_rate, mode)
        return f"文件已存储到{path}"
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def modelAnalysis(model_path, config_path, cluster_model_path, device, enhance):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        model = Svc(model_path.name, config_path.name, device=device if device != "Auto" else None,
                    cluster_model_path=cluster_model_path.name if cluster_model_path != None else "",
                    nsf_hifigan_enhance=enhance)
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"模型已加载到{device_name}\n"
        if cluster_model_path is None:
            msg += "聚类模型未加载\n"
        else:
            msg += f"聚类模型已加载\n"
        msg += "模型音色：\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices=spks, value=spks[0]), msg
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def modelUnload():
    global model
    if model is None:
        return sid.update(choices=[], value=""), "模型未卸载"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices=[], value=""), "模型已卸载"


def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num,
          lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold):
    global model
    try:
        if input_audio is None:
            raise gr.Error("无音频！")
        if model is None:
            raise gr.Error("无模型！")
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,
                                       pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key,
                                       cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        try:
            timestamp = str(int(time.time()))
            filename = sid + "_" + timestamp + ".wav"
            output_file = os.path.join("./results", filename)
            soundfile.write(output_file, _audio, model.target_sample, format="wav")
            return f"文件已存储到results/{filename}", (model.target_sample, _audio)
        except Exception as e:
            if debug: traceback.print_exc()
            return f"文件未存储", (model.target_sample, _audio)
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def tts_func(_text, _rate, _voice):
    # voice = "zh-CN-XiaoyiNeural"
    # voice = "zh-CN-YunxiNeural"
    voice = "zh-CN-YunxiNeural"
    if _voice == "女": voice = "zh-CN-XiaoyiNeural"
    output_file = _text[0:10] + ".wav"
    if _rate >= 0:
        ratestr = "+{:.0%}".format(_rate)
    elif _rate < 0:
        ratestr = "{:.0%}".format(_rate)

    p = subprocess.Popen("edge-tts " +
                         " --text " + _text +
                         " --write-media " + output_file +
                         " --voice " + voice +
                         " --rate=" + ratestr
                         , shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    p.wait()
    return output_file


def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)


def vc_fn2(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num,
           lgr_num, text2tts, tts_rate, tts_voice, F0_mean_pooling, enhancer_adaptive_key, cr_threshold):
    text2tts = text_clear(text2tts)
    output_file = tts_func(text2tts, tts_rate, tts_voice)

    sr2 = 44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2 = text2tts[0:10] + "_44k" + ".wav"
    wavfile.write(save_path2, sr2,
                  (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                  )

    sample_rate, data = gr_pu.audio_from_file(save_path2)
    vc_input = (sample_rate, data)

    a, b = vc_fn(sid, vc_input, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num,
                 lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return a, b


def debug_change():
    global debug
    debug = debug_button.value


with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            font=["Source Sans Pro", "Arial", "sans-serif"],
            font_mono=['JetBrains Mono', "Consolas", 'Courier New']),
) as app:
    with gr.Tabs():
        with gr.TabItem("so-vite-svc-lite"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    model_path = gr.File(label="选择模型文件")
                    config_path = gr.File(label="选择配置文件")
                    cluster_model_path = gr.File(label="选择聚类模型文件（可选）")
                    device = gr.Dropdown(label="CPU or GPU", choices=["Auto", *cuda.keys(), "cpu"],
                                         value="Auto")
                    enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强（默认否）",
                                          value=False)
                with gr.Column():
                    model_load_button = gr.Button(value="加载模型", variant="primary")
                    model_unload_button = gr.Button(value="卸载模型", variant="primary")
                    sid = gr.Dropdown(label="音色")
                    sid_output = gr.Textbox(label="Output Message")

            with gr.Row(variant="panel"):
                with gr.Column():
                    auto_f0 = gr.Checkbox(label="是否使用自动F0预测（默认否）", value=False)
                    F0_mean_pooling = gr.Checkbox(label="是否使用F0平均池化（默认否）", value=False)
                    vc_transform = gr.Number(label="变调（整数）", value=0)
                    cluster_ratio = gr.Number(label="cluster_ratio", value=0)
                    slice_db = gr.Number(label="slice_db", value=-40)
                    noise_scale = gr.Number(label="noise_scale", value=0.4)
                with gr.Column():
                    pad_seconds = gr.Number(label="pad_seconds", value=0.5)
                    cl_num = gr.Number(label="cl_num", value=0)
                    lg_num = gr.Number(label="lg_num", value=0)
                    lgr_num = gr.Number(label="lgr_num", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="enhancer_adaptive_key", value=0)
                    cr_threshold = gr.Number(label="F0过滤阈值（请打开F0_mean_pooling，减小cr_threshold可减少跑调，但哑音增多)",
                                             value=0.05)
            with gr.Tabs():
                with gr.TabItem("音频转音频"):
                    vc_input3 = gr.Audio(label="选择音频")
                    vc_submit = gr.Button("音频转换", variant="primary")
                with gr.TabItem("文字转音频"):
                    text2tts = gr.Textbox(label="输入文字（请打开auto_f0）")
                    tts_rate = gr.Number(label="tts语速", value=0)
                    tts_voice = gr.Radio(label="性别", choices=["男", "女"], value="男")
                    vc_submit2 = gr.Button("文字转换", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                debug_button = gr.Checkbox(label="debug模式", value=debug)
        vc_submit.click(vc_fn,
                        [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
                         cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold],
                        [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2,
                         [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
                          cl_num, lg_num, lgr_num, text2tts, tts_rate, tts_voice, F0_mean_pooling,
                          enhancer_adaptive_key, cr_threshold], [vc_output1, vc_output2])
        debug_button.change(debug_change, [], [])
        model_load_button.click(modelAnalysis, [model_path, config_path, cluster_model_path, device, enhance],
                                [sid, sid_output])
        model_unload_button.click(modelUnload, [], [sid, sid_output])
    app.launch()
