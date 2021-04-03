#将视频转换为一帧一帧的图片，只需传入两个参数，视频读取路径和图片输出路径，文件夹要提前建好。
import argparse
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description = 'split a video into frames and save to a folder')
    parser.add_argument('video_path', help='video full path')
    parser.add_argument('out_dir', help='folder for frames storage')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    video_path =args.video_path
    out_dir = args.out_dir
    if video_path:
        my_video = mmcv.VideoReader(video_path)
    if out_dir:
        my_video.cvt2frames(out_dir)


if __name__ == '__main__':
    main()