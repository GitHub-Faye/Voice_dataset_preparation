import argparse
import os
import shutil
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

import logging

vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    onnx=True,
    trust_repo=True,
)#torch.hub.load 将会返回加载的声音活动检测模型对象（vad_model）和相关的实用工具（utils）。这样，你就可以在代码中使用 vad_model 进行声音活动检测，而 utils 则可能包含了一些辅助函数或工具类，用于处理模型的输入、输出或其他任务。

(get_speech_timestamps, _, read_audio, *_) = utils#(get_speech_timestamps, _, read_audio, *_): 这是一个元组解包的语法。它会从 utils 中按顺序提取元素，并将其赋值给等号左侧的变量。在这里，通过解包，从 utils 中获取了具体的函数。


def get_stamps(
    audio_file, min_silence_dur_ms: int = 700, min_sec: float = 2, max_sec: float = 12
):
    """
    min_silence_dur_ms: int (ミリ秒):
        这个毫秒数以上判断为无声。相反，在该秒数以下的无声区间中不被划分。调小的话，声音太小，调大的话，每个声音都太长。根据数据集大概需要调整。
    min_sec: float (秒):
        能接受的最小说话秒数
    max_sec: float (秒):
        能接受的最大说话秒数
    """

    sampling_rate = 16000  # 只支持16khz或8khz

    min_ms = int(min_sec * 1000) #调整时间单位

    wav = read_audio(audio_file, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_s=max_sec,
    )#获得时间戳信息

    return speech_timestamps


def split_wav(
    audio_file,
    target_dir="raw",
    min_sec=2,
    max_sec=12,
    min_silence_dur_ms=700,
):
    margin = 200  # 以毫秒为单位，让声音前后留有余地
    #获取音频时间戳
    speech_timestamps = get_stamps(
        audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )

    data, sr = sf.read(audio_file) #返回读取的音频数据和采样率

    total_ms = len(data) / sr * 1000 #音频时长

    file_name = os.path.basename(audio_file).split(".")[0] #文件名
    os.makedirs(target_dir, exist_ok=True)#创建目录

    total_time_ms = 0

    # 根据时间戳进行分割，保存到文件中
    for i, ts in enumerate(speech_timestamps):
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        sf.write(os.path.join(target_dir, f"{file_name}-{i}.wav"), segment, sr)
        total_time_ms += end_ms - start_ms

    return total_time_ms / 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#用于创建命令行解析器。它的作用是帮助你定义命令行界面，解析用户输入的命令行参数，并将其转换为易于处理的数据结构。
    parser.add_argument(#通过 add_argument 方法，你可以定义需要从命令行中获取的参数。这可以是位置参数（没有前缀，根据位置确定），也可以是选项参数（有前缀，比如 --input
        "--min_sec", "-m", type=float, default=2, help="Minimum seconds of a slice" #指定默认切割最小秒
    )
    parser.add_argument(
        "--max_sec", "-M", type=float, default=12, help="Maximum seconds of a slice"  #指定默认切割最大秒
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="inputs",
        help="Directory of input wav files",  #指定输入目录
    )
    parser.add_argument(  #指定输出目录
        "--output_dir",
        "-o",
        type=str,
        default="raw",
        help="Directory of output wav files",
    )
    parser.add_argument(  #沉默时间分割点
        "--min_silence_dur_ms",
        "-s",
        type=int,
        default=700,
        help="Silence above this duration (ms) is considered as a split point.",
    )
    args = parser.parse_args()#拿取参数

    input_dir = args.input_dir
    output_dir = args.output_dir
    min_sec = args.min_sec
    max_sec = args.max_sec
    min_silence_dur_ms = args.min_silence_dur_ms

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个日志处理器并将其添加到根记录器
    console_handler = logging.StreamHandler()
    logging.getLogger('').addHandler(console_handler)

    
    # print(args.input_dir)
    wav_files = Path(input_dir).glob("**/*.wav")#Path(input_dir).glob("**/*.wav") 返回一个生成器，该生成器会产生所有指定目录及其子目录中以 .wav 结尾的文件的 Path 对象。你可以使用这个生成器来迭代获取所有匹配的文件路径。
    wav_files = list(wav_files)#通过 list() 函数将生成器对象转换为列表，你可以一次性获取生成器中的所有元素
    logging.info(f"Found {len(wav_files)} wav files.")
    logging.info(f"Found {len(wav_files)} wav files.")#输出找到的文件长度
    if os.path.exists(output_dir):  #如果目录存在 清空目录
        logging.warning(f"Output directory {output_dir} already exists, deleting...")
        shutil.rmtree(output_dir)# 使用 shutil 模块的 rmtree 函数，递归地删除指定目录及其所有内容。这相当于清空输出目录。

    total_sec = 0 #计算总共花费时长
    for wav_file in tqdm(wav_files):  #显示进度条的迭代
        time_sec = split_wav(
            audio_file=str(wav_file),
            target_dir=output_dir,
            min_sec=min_sec,
            max_sec=max_sec,
            min_silence_dur_ms=min_silence_dur_ms,
        )
        total_sec += time_sec #增加单次切分的时长

    logging.info(f"Slice done! Total time: {total_sec / 60:.2f} min.")
