import imageio.v2 as imageio
import os
img_dir = './imgToGif'
all_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

# Title img to int 
all_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0 ]))

with imageio.get_writer('movie.gif', mode='I') as writer:
    for filename in all_files:
        image = imageio.imread(filename)
        writer.append_data(image)
        # remove files
        os.remove(filename)
