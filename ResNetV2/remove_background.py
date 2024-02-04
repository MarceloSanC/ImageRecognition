from rembg import remove
import easygui
from PIL import Image

input_path = easygui.fileopenbox(title='Select image file')
output_path = easygui.filesavebox(title='Save file to..')

input = Image.open(input_path)
output = remove(input)
output.save(output_path)