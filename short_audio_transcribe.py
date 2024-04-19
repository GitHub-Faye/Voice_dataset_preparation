import whisper
import os
import json
import torchaudio
import argparse
import torch
from config import config
import sys
from common.log import logger
import yaml
from common.constants import Languages
with open('config.yml', mode="r", encoding="utf-8") as f:
    configyml=yaml.load(f,Loader=yaml.FullLoader)
#该变量 configyml 现在是一个包含“config.yml”文件中的数据的 Python 字典。

#model_name = configyml["dataset_path"].replace("Data/","")


lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }
def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return lang, result.text
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default="raw")
    parser.add_argument("--output_file", "-o", type=str, default="esd.list")
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="Bert")

    args = parser.parse_args()


    output_file = args.output_file
    model_name = args.speaker_name
    speaker_name = args.speaker_name
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model_type = args.model_type
    if args.languages == "CJE":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
            "ko": "KO|",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "ko": "KO|",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "ZH|",
            "ko": "KO|",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = args.input_dir
    #parent_dir=config.resample_config.in_dir
    #parent_dir = parent_dir.replace("/audios","")
    print(parent_dir)
    speaker = model_name
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    # 2023/4/21: Get the target sampling rate
    '''with open(config.train_ms_config.config_path,'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']'''
    processed_files = 0

    wav_files = [
        os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith(".wav")
    ]#在执行此行代码后，该 wav_files 列表将包含 input_dir .然后，此列表可用于进一步处理，例如加载和分析这些文件中的音频数据。
    if os.path.exists(output_file):#总之，如果 output_file 存在，代码会记录警告，通过使用“.bak”扩展名重命名现有文件来创建现有文件的备份，并删除任何同名的现有备份文件。这是防止覆盖现有重要文件和提供备份机制的常见做法。
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        if os.path.exists(output_file + ".bak"):
            logger.warning(f"{output_file}.bak exists, deleting...")
            os.remove(output_file + ".bak")
        os.rename(output_file, output_file + ".bak")

    # language_id = Languages.JP.value    

    for i, wavfile in enumerate(wav_files):
        # try to load file as audio
        # if wavfile.startswith("processed_"):
        #     continue
        try:
            # wav, sr = torchaudio.load(parent_dir + "/" + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True,
            #                           channels_first=True)
            # wav = wav.mean(dim=0).unsqueeze(0)
            # if sr != target_sr:
            #     wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
            # if wav.shape[1] / sr > 20:
            #     print(f"{wavfile} too long, ignoring\n")
            #save_path = parent_dir+"/"+ speaker + "/" + f"ada_{i}.wav"
            # torchaudio.save(save_path, wav, target_sr, channels_first=True)
            # transcribe text
            save_path = parent_dir+"/"+ speaker + f"-{i}.wav"
            file_name = os.path.basename(wavfile)
            lang, text = transcribe_one(wavfile)
            if lang not in list(lang2token.keys()):
                print(f"{lang} not supported, ignoring\n")
                continue

            #text = "ZH|" + text + "\n"
            if model_type == "Bert":
                text = f"{file_name}|{speaker_name}|JP|{text}\n"
            else:
                #vocal_path|speaker_name|language|text
                text = f"{save_path}|{speaker_name}|ja|{text}\n"
            #text = f"./Data/{model_name}/wavs/{wavfile}|" + f"{model_name}|" +lang2token[lang] + text + "\n"
            speaker_annos.append(text)
            
            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")
        except Exception as e:
            print(e)
            continue

    # # clean annotation
    # import argparse
    # import text
    # from utils import load_filepaths_and_text
    # for i, line in enumerate(speaker_annos):
    #     path, sid, txt = line.split("|")
    #     cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
    #     cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    #     speaker_annos[i] = path + "|" + sid + "|" + cleaned_text
    # write into annotation
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)

