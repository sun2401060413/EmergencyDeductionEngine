import imageio
import os

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def get_imgs_list(path):
    filelist = os.listdir(path)
    return [os.path.join(path, elem) for elem in filelist]

def gen_gif(path, output, duration = 0.3):
    image_list = get_imgs_list(path)
    gif_name = os.path.join(path, output)
    create_gif(image_list, gif_name, duration)

#
def main():
    path = r"D:\Project\EmergencyDeductionEngine\docs\figs\imgs"
    output = 'time_space_evolution_devolution_three_pts.gif'
    gen_gif(path, output)


if __name__ == '__main__':
    main()
