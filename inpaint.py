import gradio as gr
import os, sys, requests

IMAGE_0_ADDR = 'https://github.com/GastonMazzei/escher-project-website/blob/main/0.png?raw=true'
IAGE_O_MASK_ADDR = 'https://github.com/GastonMazzei/escher-project-website/blob/main/0-mask.png?raw=true'

img = gr.processing_utils.encode_url_or_file_to_base64(IMAGE_0_ADDR)
print('finished fetching img')
mask = gr.processing_utils.encode_url_or_file_to_base64(IAGE_O_MASK_ADDR)
print('finished fetching mask')



r = requests.post(url='https://hf.space/gradioiframe/Epoching/GLIDE_Inpaint/+/api/predict/',
	json={"data": [img,mask,sys.argv[1]]})
print(f'finished requesting')

with open("resp_text.txt", "w") as file:
    file.write(r.text)


s = r.json()['data'][0]
temp = gr.processing_utils.decode_base64_to_file(s)
os.system(f'cp "{temp.name}" result.png')
print(f'finished collecting the temporal converted result')
