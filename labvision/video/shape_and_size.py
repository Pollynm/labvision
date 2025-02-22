import ffmpeg
import subprocess
import os

__all__ = ["crop", "dimensions", "resize"]


def crop(
        filename,
        crop,
        out_name=None,
        speed='superfast',
        bit_rate='20000k'):

    if out_name is None:
        core, ext = os.path.splitext(filename)
        if ext == '.MP4':
            ext = '.mp4'
        out_name = core + '_crop' + ext
    width = crop.xmax - crop.xmin
    height = crop.ymax - crop.ymin
    stream = ffmpeg.input(filename)
    stream = ffmpeg.crop(
        stream,
        crop.xmin,
        crop.ymin,
        width,
        height)
    stream = ffmpeg.output(
        stream,
        out_name,
        preset=speed,
        video_bitrate=bit_rate)
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, quiet=True)


def dimensions(filename):
    command_w = ['ffprobe',
                '-v', 'error',
                '-show_entries',
                'stream=width',
                '-of',
                'csv=p=0:s=x',
                filename]
    command_h = ['ffprobe',
                 '-v', 'error',
                 '-show_entries',
                 'stream=height',
                 '-of',
                 'csv=p=0:s=x',
                 filename]
    width_out = subprocess.check_output(command_w)
    width = int(width_out.decode("utf-8"))
    height_out = subprocess.check_output(command_h)
    height = int(height_out.decode("utf-8"))
    return width, height


def resize(input_filename, output_filename, scale=0.5):
    width, height = dimensions(input_filename)
    new_width = round(width * scale / 2) * 2
    new_height = round(height * scale / 2) * 2

    command = ['ffmpeg',
               '-i',
               input_filename,
               '-vf',
               'scale={}:{}'.format(new_width, new_height),
               output_filename]
    subprocess.call(command)
