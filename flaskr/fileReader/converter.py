from PIL import Image
import pandas as pd
def convert(file):
    im = Image.open(file, 'r')
    width, height = im.size
    im.thumbnail((28, 28), Image.ANTIALIAS)
    pixel_values = list(im.getdata())
    res = []
    for x in pixel_values:
    	if type(x) is int:
    		res.append(x)
    	else:
    		res.append(255 - (x[0] + x[1] + x[2]) // 3)
    print(res)
    d = {}
    for x in range (784):
    	d['pixel' + str(x)] = [res[x]]
    return pd.DataFrame(d, columns = ['pixel' + str(x) for x in range(784)])
