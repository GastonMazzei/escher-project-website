
import gradio as gr

import os, sys, requests

if sys.argv[1]==1:
	r = requests.post(url='https://hf.space/gradioiframe/valhalla/glide-text2im/+/api/predict/', json={"data": ["a potatoe"]})
else:
	r = requests.post(url='https://hf.space/gradioiframe/valhalla/glide-text2im/+/api/predict/', json={"data": [sys.argv[1]]})

s = r.json()['data'][0]

# To convert the base64 image returned by the API to an image tmp file object
temp = gr.processing_utils.decode_base64_to_file(s)

os.system(f'cp "{temp.name}" image.png')




