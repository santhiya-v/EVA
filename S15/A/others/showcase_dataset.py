
from PIL import Image

if __name__ == "__main__":
    new_im = Image.new('RGBA', (1460, 120))
    
    x_offset = 0
    for i in range(1,13):
        im = Image.open(f'/home/santhiya/Work/projects/EVA/S15/showcase_images/depth/{i}.jpg')
        im = im.resize((100,100))
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] + 20

    new_im.show()
    new_im.save('/home/santhiya/Work/projects/EVA/S15/showcase_images/depth.png')