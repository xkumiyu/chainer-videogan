import argparse
import os

import ffmpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Directory of input video files.')
    parser.add_argument('output', help='Directory of output video files.')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    all_files = os.listdir(args.input)
    video_files = [f for f in all_files if ('mp4' in f)]

    for video_file in video_files:
        in_file = os.path.join(args.input, video_file)
        out_file = os.path.join(args.output, video_file)

        ff = ffmpy.FFmpeg(
            inputs={in_file: None},
            outputs={out_file: [
                '-map', '0:0',
                # '-vcodec', 'copy',
                '-s', '64x64',
                '-frames', '32']})
        ff.run()


if __name__ == '__main__':
    main()
