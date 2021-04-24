# convert between video and frame by opencv or mmcv
# video pass filename frames pass dir
# python videobreak.py "c:\\users\\dell\\desktop\\video\\dir" "C:\\Users\\dell\\Desktop\\video\file.avi" tovideo
# python videobreak.py "c:\\users\\dell\\desktop\\video\\file.avi" "C:\\Users\\dell\\Desktop\\video\dir" toframes
import os


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='video')
    parser.add_argument('primitive', help='video pass filename and frames pass dir')
    parser.add_argument('destination', help='video pass filename and frames pass dir')
    parser.add_argument('kind', help='either tovideo or toframes')
    parser.add_argument('--manner', default='mmcv', help='either mmcv or opencv')
    args = parser.parse_args()
    return args


def videobreak():
    args = parse_args()
    primitive = args.primitive
    destination = args.destination
    kind = args.kind
    manner = args.manner

    def mmcv2video():
        """
        can support both mp4 and avi format, when mp4 format,
        please ignore error: OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
        """
        import mmcv
        mmcv.frames2video(primitive, destination)

    def mmcv2frames():
        import mmcv
        my_video = mmcv.VideoReader(primitive)
        my_video.cvt2frames(destination)

    def opencv2video():
        import cv2
        videoformat = os.path.splitext('C:\\Users\\dell\\Desktop\\video\\a.mp4',)[1][1:]
        filelist = os.listdir(primitive)
        filelistfull = [os.path.join(primitive,i) for i in filelist]
        sample = cv2.imread(filelistfull[0])
        size = (sample.shape[1],sample.shape[0])
        fps = 24
        if videoformat == 'avi':
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videoWriter = cv2.VideoWriter(destination, fourcc, fps, size)
            for i in filelistfull:
                frame = cv2.imread(i)
                videoWriter.write(frame)
            videoWriter.release()
        elif videoformat == 'mp4':
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(destination, fourcc, fps, size)
            # videoWriter = cv2.VideoWriter(destination, 0x00000021, fps, size)
            # ignore the errror: OpenCV: FFMPEG: tag 0x00000021/'!???' is not found (format 'mp4 / MP4 (MPEG-4 Part 14)')'
            for i in filelistfull:
                frame = cv2.imread(i)
                videoWriter.write(frame)
            videoWriter.release()

    def opencv2frames():
        import cv2
        # 抽取帧图片，并保存到指定路径
        video = cv2.VideoCapture()
        if not video.open(primitive):
            print("can not open the video")
            exit(1)
        count = 1
        index = 1
        frequency = 1
        os.mkdir(destination)
        while True:
            _, frame = video.read()
            if frame is None:
                break
            if count % frequency == 0:
                save_path = "{}/{:>03d}.jpg".format(destination, index)
                cv2.imwrite(save_path, frame)
                index += 1
            count += 1
        video.release()
        # 打印出所提取帧的总数
        print("Totally save {:d} pics".format(index - 1))

    if kind == 'tovideo':
        if manner == 'mmcv':
            mmcv2video()
        elif manner == 'opencv':
            opencv2video()
    elif kind == 'toframes':
        if manner == 'mmcv':
            mmcv2frames()
        elif manner == 'opencv':
            opencv2frames()


if __name__ == '__main__':
    videobreak()
