import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter
from PIL import Image
import sklearn.cluster
from statistics import stdev
from scipy import ndimage, misc
from cv2 import COLOR_RGB2Luv
import cv2
import pickle
from argparse import ArgumentParser
parser = ArgumentParser(description="",usage='use "python3 %(prog)s --help" for more information')
parser.add_argument('--img_path','-i')
parser.add_argument('--output_dir','-o')
parser.add_argument('--model_file','-m', help = "pretrained model file")
args = parser.parse_args()
img_path = args.img_path
output_dir = args.output_dir
model_file = args.model_file


def pairwise_dist(Y):
    dc = {}
    for i in range(len(Y)-1):
        for j in range(i+1,len(Y)):
            a = np.array(Y[i])
            b = np.array(Y[j])
            dist = np.linalg.norm(a-b)
            dc[(i,j)] = dist
    return dc

def pairwise_weight(Y):
    dc = {}
    for i in range(len(Y)-1):
        for j in range(i+1,len(Y)):
            dc[(i,j)] = (Y[i]+Y[j])/2
    return dc

T = [[0.31399022, 0.15537241, 0.01775239],[0.63951294, 0.75789446, 0.10944209],[0.04649755, 0.08670142, 0.87256922]]
T = np.matrix(T).T
T_inv = np.linalg.inv(T)

S_protanopia = np.matrix([[0,0,0],[1.05118294, 1, 0],[-0.05116099,0,1]]).T

transform_protanopia = T_inv * S_protanopia * T 

T_protanopia = S_protanopia * T 



def transform_color(color, axis = -1):
    # print(color)
    x = transform_protanopia * np.matrix(color).T
    x = list(np.array(x.flatten())[0,:])
    return x


def color_blind_converter(img_path):
    initial_img = Image.open(img_path)
    initial_img.load()
    img_arr = np.array(initial_img) / 255.0

    Y = []
    for i in range(img_arr.shape[0]):
        ys = []
        for j in range(img_arr.shape[1]):
            x = img_arr[i,j,:]
            y = transform_color(x)
            ys.append(y)
        Y.append(ys)
    Y1 = np.array(Y)
    im = Image.fromarray(np.uint8(Y1*255))
    return initial_img,im

def process_one_file(file,out_file):
    try:
        img1,img2 = color_blind_converter(file)
        img2.save(out_file)
        return 0,out_file
    except:
        print("black and white: ",file)
        return 1,out_file

def extract_key_colors(img_path,prefix,output_dir,k):
    if not os.path.exists(output_dir):
        os.system("mkdir -p "+output_dir)

    initial_img = Image.open(img_path)
    initial_img.load()
    X = np.array(initial_img).flatten()
    X = X.astype('int32').reshape(int(len(X)/3),3)


    c = sklearn.cluster.KMeans(k, random_state=0)

    clusters = c.fit(X)

    colors = c.cluster_centers_
    dc_cnt = Counter(c.labels_)
    count_list = [dc_cnt[i]   for i in range(k)]
    colors_bl =[transform_color(x) for x in colors]
    def plot_colors(colors,output_dir,prefix):

        n = len(colors)
        plt.pie([1/n]*n, colors = np.array(colors)/255,labels = list(range(n)))

        plt.savefig(output_dir+'/'+prefix+'.pdf')
        plt.close()
        return

    plot_colors(colors,output_dir,prefix+'_og')
    plot_colors(colors_bl,output_dir,prefix+'_cb')
    df1 = pd.DataFrame(colors,columns = ['R','G','B'])
    df1['count'] = count_list
    df2 = pd.DataFrame(colors_bl,columns = ['R','G','B'])
    df2['count'] = count_list
    df1.to_csv(output_dir+'/'+prefix+'_og.csv')
    df2.to_csv(output_dir+'/'+prefix+'_cb.csv')
    return 



def load_rgb_luv(img_path):

    initial_img = Image.open(img_path)
    initial_img.load()
    rgb1 = np.array(initial_img,'float32')/255  ### opensv takes rgb 0-1
    luv1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2Luv)

    return rgb1,luv1

def normalize_max_min(arr):
    arr = np.array(arr)
    arr = (arr - arr.min())/(arr.max()-arr.min())
    return arr

def calculate_features(img_path,img_path_cb,bins = 64, thresh = 0.01):


    ## calculate bin values for orginal image
    rgb_og,luv_og = load_rgb_luv(img_path)
    lumin_og = luv_og[:,:,0]
    lumin_og_1dnorm = normalize_max_min(np.ndarray.flatten(lumin_og))
    H_og = np.histogram(lumin_og_1dnorm,bins)[0]
    H_og_norm = (H_og/H_og.sum())
    ## calculate bin values for color blinded image
    rgb_cb,luv_cb = load_rgb_luv(img_path_cb)
    lumin_cb = luv_cb[:,:,0]
    lumin_cb_1dnorm = normalize_max_min(np.ndarray.flatten(lumin_cb))
    H_cb = np.histogram(lumin_cb_1dnorm,bins)[0]
    H_cb_norm = (H_cb/H_cb.sum())
    ## calculate histogram difference
    gh = ((H_og_norm-H_cb_norm)**2).sum()**0.5


    ### calculate edge difference
    result_og = normalize_max_min(ndimage.sobel(lumin_og))
    e_og = result_og.mean()
    result_cb = normalize_max_min(ndimage.sobel(lumin_cb))
    e_cb = result_cb.mean()
    ge = abs((e_og - e_cb) / e_og)


    ### calculate by pixel rgb color difference
    

    dij = ((rgb_og-rgb_cb)**2).sum(2)
    cij = (dij>thresh).astype(int)
    B = cij.sum()
    gp = (dij * cij).sum()/B
    return gh,ge,gp


# img_path = "paper_image/paper1_imgs/-003.jpeg"
# output_dir = "score_test/"
# model_file = 'finalized_model.sav'


os.system("mkdir -p " + output_dir  )
prefix = img_path.split('/')[-1].split('.')[0]
img_path_cb = output_dir + '/'+prefix+"_cb.jpeg"
k = 7
process_one_file(img_path,img_path_cb)
extract_key_colors(img_path,prefix,output_dir,k)

kc_og = output_dir+'/'+prefix+'_og.csv'
kc_cb = output_dir+'/'+prefix+'_cb.csv'

#### pair wise distance
dist_og = []
dist_cb = []
weights = []
df_og = pd.read_csv(kc_og)
df_cb = pd.read_csv(kc_cb)
Y_og = df_og.values[:,1:-1]
Y_cb = df_cb.values[:,1:-1]
weight = df_og.values[:,-1]
dc1 = pairwise_dist(Y_og)
dc2 = pairwise_dist(Y_cb)
dc3 = pairwise_weight(weight)
dist_og.extend(list(dc1.values()))
dist_cb.extend(list(dc2.values()))
weights.extend(list(dc3.values()))

dist_og = np.array(dist_og)
dist_cb = np.array(dist_cb)
weights = np.array(weights)
dist_shrink = dist_cb - dist_og
dist_shrink_weighted = (dist_shrink*weights).reshape(1,21).sum(1)/(weights).reshape(1,21).sum(1)
dist_2d = dist_shrink.reshape(1,21)

### calculate gh, ge, gp

gh_list =[]
ge_list = []
gp_list = []
gh,ge,gp = calculate_features(img_path,img_path_cb,bins = 64, thresh = 0.01)
gh_list.append(gh)
ge_list.append(ge)
gp_list.append(gp)

X = pd.DataFrame({'x1':dist_2d.mean(1),
                  'x2':dist_2d.min(1),
                         'x4':dist_shrink_weighted,
                         'x5':gh_list,
                         'x6':ge_list,
                         'x7':gp_list,
                  }).values


clf = pickle.load(open(model_file, 'rb'))
y_pred = clf.predict(X)
y_pred = y_pred[0]


if y_pred:
    s = "Input image %s is not CVD friendly"%img_path
    print(s)
    with open(output_dir+'/prediction.txt','w') as f:
        f.write(s)
else:
    s = "Input image %s is CVD friendly"%img_path
    print(s)
    with open(output_dir+'/prediction.txt','w') as f:
        f.write(s)





        





