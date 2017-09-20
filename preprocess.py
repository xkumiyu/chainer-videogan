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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Directory of input video files.')
    parser.add_argument('output', help='Directory of output image files and list file.')
    parser.add_argument('--tmp', default='/tmp', help='Temporary directory')
    parser.add_argument('--frame_size', default=32)
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if not os.path.isdir(args.tmp):
        os.mkdir(args.tmp)

    all_files = os.listdir(args.input)
    video_files = [f for f in all_files if ('mp4' in f)]

    video_list = []
    for video_file in video_files:
        name, _ = os.path.splitext(video_file)
        tmp_dir = os.path.join(args.tmp, name)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        in_file = os.path.join(args.input, video_file)
        tmp_file = os.path.join(tmp_dir, '%06d.png')

        video2image(in_file, tmp_file)
        n_videos = len(os.listdir(tmp_dir)) // args.frame_size

        for i in range(n_videos):
            out_dir = os.path.join(args.output, '{}_{}'.format(name, i))
            video_list.append(out_dir)

            if os.path.isdir(out_dir):
                continue
            os.mkdir(out_dir)
            for j in range(i * args.frame_size, (i + 1) * args.frame_size):
                src_file = os.path.join(tmp_dir, '{:06d}.png'.format(j + 1))
                shutil.move(src_file, out_dir)

        shutil.rmtree(tmp_dir)

    list_file = os.path.join(args.output, 'list.txt')
    with open(list_file, 'w') as f:
        for row in video_list:
            f.write(row + '\n')


if __name__ == '__main__':
    main()
