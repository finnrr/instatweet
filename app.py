import GetOldTweets3 as got
import instagram_explore as ie
import re
import requests
from io import BytesIO
import numpy as np
import cv2 
import io
#import cvlib as cv
#import freetype
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
#import textwrap








import base64
from io import BytesIO


from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates')



def get_instagram_urls(search_term: str):
    ''' fetch a list of instagram pictures urls based on seach term '''
    #res = ie.tag(search_term)
    #data, cursor = ie.tag('donald', res.cursor)
    return ie.tag_images(search_term).data[:5]

def clean_tweet(tweet_text):
    ''' clean the tweets of links, hashtags and @ tags, return first sentence '''
    result = re.sub(r"http\S+", "", tweet_text)
    result = re.sub(r"#\S+", "", result)
    result = re.sub(r"@\S+", "", result)
    result = result.partition('.')[0] + '.'
    result = result.strip()
    result = result.capitalize()
    return result

def get_tweets(search_term: str, num_of_tweets=10):
    ''' fetch a list of tweets '''
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(search_term).setMaxTweets(num_of_tweets+20)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweet_list = []
    iterration = 0
    while len(tweet_list) < num_of_tweets:
        tweet = clean_tweet(tweets[iterration].text)
        iterration += 1
        if len(tweet) > 10:
            tweet_list.append(tweet)
        if iterration + 1 >= len(tweets):
            break   
    return tweet_list

#get_tweets('hobos')

def url_to_cv(url):
    response = requests.get(url)
    response.content
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, width=720):
    ''' resize image by width '''
    scale_percent = image.shape[1] / width
    height = int(image.shape[0] / scale_percent )
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 

# import matplotlib.pyplot as plt 
# results[1]
# image = np.array(results[1]) 
# # Convert RGB to BGR 
# open_cv_image = open_cv_image[:, :, ::-1].copy() 

# plt.imshow(resize_image(image))
# resize_image(image).shape

def coords_from_cv(image):
#    faces, confidences = cv.detect_face(image)
#    coords = []
#    try:
#        coords = faces[0][0],faces[0][1]
#    except:
#        coords = (0,0)
    return (0,0) #coords

def reduce_colors(image):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
    cartoon = cv2.bitwise_and(res2, res2, mask=edges)
    return cartoon

def cv_to_pil(image):
    image = image[:,:,::-1]
    return Image.fromarray(image)

def draw_text(tweet,image,coords):
    font = ImageFont.truetype("comicfont.ttf")
    text_size = font.getsize(tweet)
    bubble_size = (text_size[0]+20, text_size[1]+20)
    bubble = Image.new('RGBA', bubble_size, "white")
    bubble_text = ImageDraw.Draw(bubble)
    bubble_text.text((10, 10), tweet, font=font, fill='black')
    image.paste(bubble, coords)
    return image



# lines = textwrap.wrap(text, width=40)
# y_text = h
# for line in lines:
#     width, height = font.getsize(line)
#     draw.text(((w - width) / 2, y_text), line, font=font, fill=FOREGROUND)
#     y_text += height





def fetch_media(search_term):
    urls = get_instagram_urls(search_term)
    tweets = get_tweets(search_term, 5)
    return urls, tweets

# media = fetch_media('hobos')

def crunch_comic(search_results, itteration=0):
    tweets = search_results[1]
    urls = search_results[0]
    url = urls[itteration]
    tweet = tweets[itteration]
    image_cv = url_to_cv(url)
    resize_cv = resize_image(image_cv)
    coords = coords_from_cv(resize_cv)
    image_reduce_cv = reduce_colors(resize_cv)
    image_pil = cv_to_pil(image_reduce_cv)
    final_image = draw_text(tweet, image_pil, coords)
    return final_image

# crunch_comic(media, 4)


def comic_strip(search_term):
    results =  fetch_media(search_term)
    panel_0 = crunch_comic(results,0)
    panel_1 = crunch_comic(results,1)
    panel_2 = crunch_comic(results,2)
    panel_3 = crunch_comic(results,3)
    return panel_0, panel_1, panel_2, panel_3


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    args = request.form
    text1 = str(args.get('input1'))
    print(text1)
    pics = comic_strip(text1)
    pic_list = []
    for i in pics:
        buffered = BytesIO()
        i.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        pic_list.append(img_str)
    pic0, pic1, pic2, pic3 = pic_list
    pics = { "pic0":pic0, "pic1":pic1, "pic2":pic2, "pic3":pic3}
    return render_template('result.html', pics=pics)



    # def post(self, request):
    #     form = HomeForm(request.POST)
    #     if form.is_valid():
    #         text = form.cleaned_data['post']
    #         pics = comic_strip(text)
    #         form = HomeForm()
    #     pic_list = []    
    #     for i in pics:
    #         buffered = BytesIO()
    #         i.save(buffered, format="JPEG")
    #         img_str = base64.b64encode(buffered.getvalue()).decode()
    #         pic_list.append(img_str)
    #     pic0, pic1, pic2, pic3 = pic_list  
    #     args = {"form":form, "pic0":pic0, "pic1":pic1, "pic2":pic2, "pic3":pic3}
    #     return render(request, 'index.html', args)



if __name__ == '__main__':
    app.run(port=8080, debug=True)



# pic1, pic2, pic3, pic4 = comic_strip('stuff')


#todo

# results = comic_strip('iran')
# results[0]
# results[1]
# results[2]

# import textwrap
# lines = textwrap.wrap(text, width=40)
# y_text = h
# for line in lines:
#     width, height = font.getsize(line)
#     draw.text(((w - width) / 2, y_text), line, font=font, fill=FOREGROUND)
#     y_text += height
#