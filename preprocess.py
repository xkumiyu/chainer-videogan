import argparse
import os
import shutil

import ffmpy


def video2image(in_file, out_file):
    ff = ffmpy.FFmpeg(
        inputs={in_file: None},
        outputs={out_file: [
            '-vcodec', 'png',
            '-r', '25',
            '-s', '64x64']})
    ff.run()


def image2video(in_file, out_file, start_number):
    ff = ffmpy.FFmpeg(
        inputs={in_file: [
            '-framerate', '25',
            '-start_number', str(start_number)]},
        outputs={out_file: [
            '-vframes', '32',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-r', '25',
            '-s', '64x64']})
    ff.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Directory of input video files.')
    parser.add_argument('output', help='Directory of output video files.')
    parser.add_argument('-t', '--tmp', default='/tmp',
                        help='Temporary directory')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if not os.path.isdir(args.tmp):
        os.mkdir(args.tmp)

    all_files = os.listdir(args.input)
    video_files = [f for f in all_files if ('mp4' in f)]

    for video_file in video_files:
        name, _ = os.path.splitext(video_file)
        tmp_dir = os.path.join(args.tmp, name)

        in_file = os.path.join(args.input, video_file)
        tmp_file = os.path.join(tmp_dir, '%06d.png')

        # if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        video2image(in_file, tmp_file)

        n = len(os.listdir(tmp_dir)) // 32
        for i in range(n):
            out_file = os.path.join(args.output, '{}_{}.mp4'.format(name, i))
            if not os.path.isfile(out_file):
                image2video(tmp_file, out_file, i * 32)

        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()
