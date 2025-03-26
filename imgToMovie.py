import imageio.v2 as imageio
import os
import argparse
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def main(img_dir, fps):
    all_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

    # Title img to int 
    all_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    clip = ImageSequenceClip(all_files, fps=fps)
    clip.write_videofile(f'{img_dir}.mp4', codec='libx264')

    with imageio.get_writer(f'movie.gif', mode='I') as writer:
        for filename in all_files:
            image = imageio.imread(filename)
            writer.append_data(image)
            # remove files
            os.remove(filename)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to a GIF movie.')
    parser.add_argument('img_dir', type=str, help='Directory containing the images')
    # parser.add_argument('fps', type=int, help='Frames per second')
    args = parser.parse_args()
    # if args.fps is None:
    #     args.fps = 1
    fps = 2
    main(args.img_dir, fps=fps)
