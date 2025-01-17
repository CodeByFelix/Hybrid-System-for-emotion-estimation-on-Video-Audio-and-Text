#import PySimpleGUI as sg
import flet as ft
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
import seaborn as sns
import matplotlib.pyplot as plt
import whisper
import moviepy as mpy
import librosa.display
import librosa
from IPython.display import Audio
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#from tensorflow.keras.models import load_model
from keras.models import load_model
import joblib
import os
from datetime import datetime
import base64
import pyaudio
import wave

# def generatePlot (data, title, fileName):
#     emotions = list(data.keys())
#     scores = list(data.values())
    
#     plt.figure(figsize=(6,4))
#     bars = plt.bar(emotions, scores, color='skyblue')
#     plt.title(title)
#     plt.xlabel("Emotions")
#     plt.ylabel("Scores")
#     #plt.ylim(0, 1)
#     #plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), 
#                  ha='center', va='bottom', fontsize=10, color='black')
    
#     plt.savefig(fileName, format='png', dpi=300)
#     plt.close()

def generatePlot(data, title, fileName):
    # Convert data dictionary to two lists
    emotions = list(data.keys())
    scores = list(data.values())

    # Create a figure
    plt.figure(figsize=(6, 4))

    # Create a Seaborn bar plot
    ax = sns.barplot(x=emotions, y=scores)

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Scores")
    #ax.set_ylim(0, 1)  # Optional: Set y-axis limit
    #ax.set_xticklabels(emotions, rotation=45)  # Optional: Rotate x-axis labels

    # Annotate bars with their values
    for bar, score in zip(ax.patches, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                round(score, 2), ha='center', va='bottom', fontsize=10, color='black')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(fileName, format='png', dpi=300)
    plt.close()
    

def extractAudio (filePath):
    cvtVideo = mpy.VideoFileClip(filePath)
    extAudio = cvtVideo.audio
    extAudio.write_audiofile('audio.mp3')
    
def extractText (fileName):
    result = audio_to_text.transcribe(fileName)
    return result['text']

def perResize (image, percent, inter=cv2.INTER_AREA):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def emotionVideo (filePath):
    gone = False
    
    video = cv2.VideoCapture ('Video1.mp4')

    result_dic = {'angry': 0.0,
     'disgust': 0.0,
     'fear': 0.0,
     'happy': 0.0,
     'sad': 0.0,
     'surprise': 0.0,
     'neutral': 0.0}
    
    while True:
        ret, frame = video.read()
        if ret:
            if gone: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = perResize (frame, percent=20)
                #frame = cv2.resize(frame, (480, 320))
                try:
                    pred = DeepFace.analyze (frame, actions='emotion')
                except:
                    pass
                else:
                    emotion_dic = pred[0]['emotion']
                    result_dic = {key: result_dic[key] + emotion_dic[key] for key in result_dic}
                    print (emotion_dic)
                
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = perResize (frame, percent=20)
                #frame = cv2.resize(frame, (480, 320))
                try:
                    pred = DeepFace.analyze (frame)
                    gone = True
                except:
                    pass
                else:
                    emotion_dic = pred[0]['emotion']
                    result_dic = {key: result_dic[key] + emotion_dic[key] for key in result_dic}
                    print (emotion_dic)
                    otherResult = {'age': pred[0]['age'],
                                   'gender': pred[0]['dominant_gender'],
                                   'race': pred[0]['dominant_race']}
        else:
            break
    return result_dic, otherResult

def audioPreprocess (fileName):
    y, sr = librosa.load(fileName, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    dd = np.expand_dims(mfcc, axis=0)
    dd = np.expand_dims(dd, -1)
    return dd

def emotionAudio (fileName):
    audio_data = audioPreprocess (fileName)
    audio_pred = audio_model.predict (audio_data)
    audio_emotion = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']
    audio_emotion_dic = {audio_emotion[i]: round(audio_pred[0][i], 3) for i in range(0, len(audio_emotion))}
    return audio_emotion_dic

def textPreprocess (line):
    review = re.sub('[^a-zA-Z]', ' ', line) #Remove all puntuations leave only characters.
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    new_text = text_cv.transform([review]).toarray()
    return new_text

def emotionText (text):
    text_data = textPreprocess (text)
    text_pred = text_model.predict(text_data)
    text_emotion = ['angry', 'fear', 'joy', 'love', 'sad', 'surprise']
    text_emotion_dic = {text_emotion[i]: round(text_pred[0][i], 3) for i in range(0, len(text_emotion))}
    return text_emotion_dic
        

def mainWindow (page: ft.Page):
    page.title = "Hybrid"
    page.adaptive=True
    #page.bgcolor="#E4E0E1"
    page.scroll="auto"
    
    
    def filePickerResult (event):
        if event.files:
            ppp.value = "                                        "
            print(event.files)
            page.update()
            filePath = event.files[0].path
            rrr = extractFeatures (filePath)
            plotDisplay(rrr)
            #updateUI('Done Processing')
    
    def homePage (p):
        plot.visible=False
        ppp.value = ""
        videoPlot.controls.clear()
        audioPlot.controls.clear()
        textPlot.controls.clear()
        aboutLayout.visible = False
        layout.visible = True
        page.update()
    
    def aboutPage (p):
        layout.visible = False
        aboutLayout.visible = True
        page.update ()
    
    def loadFile (p):
        filePicker.pick_files(allow_multiple=False)
        
    def extractFeatures (filePath):
        updateUI("Processing 10%\nExtracting Features")
        extractAudio (filePath)
        text = extractText ('audio.mp3')
        updateUI("Processing 30%\nPredicting on Video")
        videoResult, videoOther = emotionVideo(filePath)
        updateUI ("Processing 70%\nPredicting on Audio")
        audioResult = emotionAudio ('audio.mp3')
        updateUI ("Processing 80%\nPredicting on text")
        textResult = emotionText (text)
        updateUI ("Processing 90%\nProcessing Final Result")
        finalResult = [videoResult, videoOther, audioResult, textResult]
        return finalResult
    
    def updateUI (text):
        uuu.value = text
        page.update ()
    
    def plotDisplay (result):
    #def plotDisplay ():
        
        video = result[0]
        videoOther = result[1]
        audio = result[2]
        text = result [3]
        
        # video = {'angry': 3.2,
        #          'joy': 2.3,
        #          'sad': 8.2}
        # audio = {'angry': 3.2,
        #          'joy': 2.3,
        #          'sad': 8.2}
        # text = {'angry': 3.2,
        #          'joy': 2.3,
        #          'sad': 8.2}
        
        currentTime = datetime.now()
        folderName = currentTime.strftime("%Y-%m-%d__%H-%M-%S")
        basePath = os.getcwd()
        folderPath = os.path.join(basePath, 'Processed', folderName)
        try:
            os.makedirs(folderPath)
        except FileExistsError:
            pass
        
        videoName = os.path.join (folderPath, 'video_plot.png')
        audioName = os.path.join (folderPath, 'audio_plot.png')
        textName = os.path.join (folderPath, 'text_plot.png')
        generatePlot(video, "Video Emotion Score", videoName)
        generatePlot(audio, "Audio Emotion Score", audioName)
        generatePlot(text, "text Emotion Score", textName)
        
        im1 = ft.Container(
            content = ft.Column(
            controls=[ft.Text(f"Gender: {videoOther['gender']}\nAge: {videoOther['age']}\nRace: {videoOther['race']}",
                              size=20,
                              color='blue'),
                      
                      ft.Container (
                content=ft.Image(src=videoName,
                                 width=700,
                                 height=500)
                ),
                #ft.Text("Video")
                ]
            ))
        
        im2 = ft.Container(
            content = ft.Column(
            controls=[ft.Container (
                content=ft.Image(src=audioName,
                                 width=700,
                                 height=500)
                ),
                #ft.Text("audio")
                ]
            ))
        
        im3 = ft.Container(
            content = ft.Column(
            controls=[ft.Container (
                content=ft.Image(src=textName,
                                 width=700,
                                 height=500)
                ),
                #ft.Text("text")
                ]
            ))
        
        videoPlot.controls.clear()
        audioPlot.controls.clear()
        textPlot.controls.clear()
        
        videoPlot.controls.append(im1)
        audioPlot.controls.append(im2)
        textPlot.controls.append(im3)
        plot.visible=True
        updateUI("")
        page.update()
    
    def showImage (image, vvv):
        try:
            _, buffer = cv2.imencode ('.jpg', image)
            imageByte = base64.b64encode(buffer).decode('utf-8')
            image_src = imageByte
            #image_scr = f"data:image/jpeg;base64,{imageByte}"
            # imageByte = buffer.tobytes ()
            # image_scr = f"data:image/jpeg;base64,{imageByte.decode('latin1')}"
            
            vvv.src_base64 = image_src
            
            page.update()
            
        except Exception as e:
            print (e)
    
    def runEmotion (image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            imagePred = DeepFace.analyze (image, actions='emotion')
        except ValueError:
            cv2.putText(image, "No Face",  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            emotionPred = {'angry': 0.0,
                     'disgust': 0.0,
                     'fear': 0.0,
                     'happy': 0.0,
                     'sad': 0.0,
                     'surprise': 0.0,
                     'neutral': 0.0}
        else:
            x = imagePred[0]["region"]['x']
            y = imagePred[0]["region"]['y']
            w = imagePred[0]["region"]['w']
            h = imagePred[0]["region"]['h']
            emotionPred = imagePred[0]['emotion']
            cv2.rectangle (image, (x,y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(image, imagePred[0]["dominant_emotion"],  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, emotionPred
    
    def runEmotion1 (image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            imagePred = DeepFace.analyze (image)
        except ValueError:
            cv2.putText(image, "No Face",  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            emotionPred = {'angry': 0.0,
                     'disgust': 0.0,
                     'fear': 0.0,
                     'happy': 0.0,
                     'sad': 0.0,
                     'surprise': 0.0,
                     'neutral': 0.0}
        else:
            x = imagePred[0]["region"]['x']
            y = imagePred[0]["region"]['y']
            w = imagePred[0]["region"]['w']
            h = imagePred[0]["region"]['h']
            emotionPred = imagePred[0]['emotion']
            cv2.rectangle (image, (x,y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(image, imagePred[0]["dominant_emotion"],  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, emotionPred
    
    
    def realTime (p):
        global stopVideo
        stopVideo = False
        updateUI('Preparing Video')
        
        
        audioFileName = 'output.wav'
        chunk = 10240
        sampleFormat = pyaudio.paInt16
        channels = 1
        rate = 44100
        
        audio = pyaudio.PyAudio()
        
        stream = audio.open (format=sampleFormat,
                             channels=channels,
                             rate=rate,
                             input=True,
                             frames_per_buffer=chunk)
        frames = []
        
        
        result_dic = {'angry': 0.0,
                 'disgust': 0.0,
                 'fear': 0.0,
                 'happy': 0.0,
                 'sad': 0.0,
                 'surprise': 0.0,
                 'neutral': 0.0}
        
        cam = cv2.VideoCapture (0)
        #createVidioWidget ()
        vvv = ft.Image(
                         width=700,
                         height=500)
        im = ft.Container(
            content = ft.Column(
            controls=[ft.Container (
                content=vvv
                ),
                #ft.Text("audio")
                ]
            ))
        videoShow.controls.clear()
        videoPlot.controls.clear()
        audioPlot.controls.clear()
        textPlot.controls.clear()
        videoShow.controls.append(im)
        plot.visible = True
        
        page.update()
        gone = False
        while cam.isOpened ():
            if not gone:
                
                ret, frame = cam.read()
                frame = cv2.flip(frame, 1)
                frame = perResize (frame, percent=70)
                showImage (frame, vvv)
                #frame, emotionPred = runEmotion1 (frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    imagePred = DeepFace.analyze (frame)
                    gone = True
                    updateUI('Video Processing...\nAudio Recording...')
                except ValueError:
                    cv2.putText(frame, "No Face",  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)
                else:
                    x = imagePred[0]["region"]['x']
                    y = imagePred[0]["region"]['y']
                    w = imagePred[0]["region"]['w']
                    h = imagePred[0]["region"]['h']
                    emotionPred = imagePred[0]['emotion']
                    cv2.rectangle (frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, imagePred[0]["dominant_emotion"],  (20,20), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 255, 0), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    showImage (frame, vvv)
                    result_dic = {key: result_dic[key] + emotionPred[key] for key in result_dic}
                    otherResult = {'age': imagePred[0]['age'],
                                   'gender': imagePred[0]['dominant_gender'],
                                   'race': imagePred[0]['dominant_race']}
                if stopVideo:
                    break
            
            else:
                ret, frame = cam.read()
                data = stream.read(chunk)
                frames.append (data)
                frame = cv2.flip(frame, 1)
                frame = perResize (frame, percent=70)
                frame, emotionPred = runEmotion (frame)
                showImage (frame, vvv)
                result_dic = {key: result_dic[key] + emotionPred[key] for key in result_dic}
                
                if stopVideo:
                    break
        
        cam.release()
        updateUI('Processing 10%\nProcessing Audio')
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        with wave.open(audioFileName, 'wb') as wf:
            wf.setnchannels (channels)
            wf.setsampwidth (audio.get_sample_size(sampleFormat))
            wf.setframerate(rate)
            wf.writeframes (b''.join(frames))
            
        videoShow.controls.clear()
        page.update()
        updateUI ('Processing 40%\nExtracting text')
        text = extractText(audioFileName)
        print (f"Extracted Text is {text}")
        updateUI ('Processing 60%\nPredicting')
        audio_dic = emotionAudio(audioFileName)
        text_dic = emotionText (text)
        updateUI ('Processing 90% \nFinal Results')
        plotDisplay([result_dic, otherResult, audio_dic, text_dic])
        
        
        
    
    def stopRealTime (p):
        global stopVideo
        stopVideo = True
    
    def audioPredict (event):
        if event.files:
            filePath = event.files[0].path
            #updateUI (filePath)
            updateUI ("Processing 10%\nExtracting Text")
            text = extractText(filePath)
            updateUI ("Processing 60%\nPredicting on Audio")
            audioResult = emotionAudio(filePath)
            updateUI ("Processing 80%\nPredicting on Text")
            textResult = emotionText (text)
            updateUI ("Processing 90%\nFinal Processing")
            
            currentTime = datetime.now()
            folderName = currentTime.strftime("%Y-%m-%d__%H-%M-%S")
            basePath = os.getcwd()
            folderPath = os.path.join(basePath, 'Processed', folderName)
            try:
                os.makedirs(folderPath)
            except FileExistsError:
                pass
            
            audioName = os.path.join (folderPath, 'audio_plot.png')
            textName = os.path.join (folderPath, 'text_plot.png')
            
            generatePlot(audioResult, "Audio Emotion Score", audioName)
            generatePlot(textResult, "text Emotion Score", textName)
            
            im2 = ft.Container(
                content = ft.Column(
                controls=[ft.Container (
                    content=ft.Image(src=audioName,
                                     width=700,
                                     height=500)
                    ),
                    #ft.Text("audio")
                    ]
                ))
            
            im3 = ft.Container(
                content = ft.Column(
                controls=[ft.Container (
                    content=ft.Image(src=textName,
                                     width=700,
                                     height=500)
                    ),
                    #ft.Text("text")
                    ]
                ))
            
            textTranscribe.value = ""
            videoPlot.controls.clear()
            audioPlot.controls.clear()
            textPlot.controls.clear()
            
            audioPlot.controls.append(im2)
            textPlot.controls.append(im3)
            plot.visible=True
            updateUI("")
            page.update()
            
    
    def audioTranscribe (event):
        if event.files:
            filePath = event.files[0].path
            updateUI ("Transcribing")
            text = extractText(filePath)
            updateUI ("")
            
            textTranscribe.value = text
            videoPlot.controls.clear()
            audioPlot.controls.clear()
            textPlot.controls.clear()
            
            plot.visible=True
            page.update()
            
    
    navbar = ft.Container(
        content= ft.Column(
            controls=[
                ft.Text("Hybrid Model For Pattern Extraction in Video, Audio and Text",
                        size=30,
                        color='white',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
                ft.Row(
                    controls=[
                        ft.TextButton(
                            content=ft.Text("Home",
                                            size=15,
                                            color='white',
                                            font_family='Arial',
                                            weight=ft.FontWeight.BOLD),
                            on_click=homePage
                            ),
                        ft.TextButton(
                            content=ft.Text("About",
                                            size=15,
                                            color='white',
                                            font_family='Arial',
                                            weight=ft.FontWeight.BOLD),
                            on_click=aboutPage
                            )
                       
                        ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    scroll="none"
                    )
                
                ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
        height=150,
        bgcolor='blue',
        #bgcolor='#EED3B1',
        alignment = ft.alignment.center,
        )
    
    loadButton = ft.TextButton (
        content=ft.Text("Load File",
                        size=30,
                        color='Blue',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
        
        on_click=loadFile
        )
    
    realTimeButton = ft.TextButton (
        content=ft.Text("Real-Time Processing",
                        size=30,
                        color='Blue',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
        on_click=realTime
        )
    
    stopRealTimeButton = ft.TextButton (
        content=ft.Text("Stop Real-Time",
                        size=30,
                        color='Blue',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
        on_click=stopRealTime
        )
    
    audioPredictButton = ft.TextButton (
        content=ft.Text("Audio Predict",
                        size=30,
                        color='Blue',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
        on_click=lambda _ : audioPredictFilePicker.pick_files(allow_multiple=False)
        )
    
    audioTranscribeButton = ft.TextButton (
        content=ft.Text("Audio Transcribe",
                        size=30,
                        color='Blue',
                        font_family='Arial',
                        weight=ft.FontWeight.BOLD
                        ),
        on_click=lambda _ : audioTranscribeFIlePicker.pick_files(allow_multiple=False)
        )
    
    buttonLayout = ft.Container(content= ft.Container(
        content=ft.Column(controls=[loadButton, realTimeButton, stopRealTimeButton,
                                    audioPredictButton, audioTranscribeButton],
                          alignment=ft.MainAxisAlignment.CENTER,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                          #height=800,
                          #scroll=ft.ScrollMode.ALWAYS
                          scroll='auto'
                          ),
        #height=400,
        alignment=ft.alignment.center,
        border=ft.border.all(color="blue", width=2),           
        border_radius=20,  
        padding = ft.Padding(top=20, right=20, bottom=20, left=20)
        ),
        padding = ft.Padding(top=50, right=50, bottom=0, left=200)
        )
    
    filePicker = ft.FilePicker (on_result=filePickerResult)
    audioPredictFilePicker = ft.FilePicker (on_result=audioPredict)
    audioTranscribeFIlePicker = ft.FilePicker (on_result=audioTranscribe)
    
    page.overlay.append(filePicker)
    page.overlay.append (audioPredictFilePicker)
    page.overlay.append (audioTranscribeFIlePicker)
       
    
    
    
    videoShow = ft.Column(
        controls=[],
        scroll='auto'
        )
    
    videoPlot = ft.Column(
        controls=[],
        scroll='auto'
        )
    audioPlot = ft.Column (
        controls=[],
        scroll='auto'
        )
    textPlot = ft.Column (
        controls=[],
        scroll='auto'
        )
    
    textTranscribe = ft.TextField (
                                color='blue',
                                read_only=True,
                                border_color=ft.colors.TRANSPARENT,
                                text_align=ft.TextAlign.JUSTIFY,
                                text_size=15,
                                multiline = True)
    transcribeContainer = ft.Container (content=textTranscribe, 
                                        alignment=ft.alignment.center,
                                        width=700,
                                        padding = ft.Padding(top=20, right=20, bottom=20, left=20))
    
    plot = ft.Container (content = ft.Container(
        content=ft.Column(
            controls=[videoShow, videoPlot, audioPlot, textPlot, transcribeContainer],
            #visible=True,
            #scroll=ft.ScrollMode.ALWAYS
            
            ),
        
        alignment=ft.alignment.center,
        border=ft.border.all(color="blue", width=2),           
        border_radius=20,
        ),
        visible=False,
        padding = ft.Padding(top=20, right=0, bottom=0, left=50)
        )
    
    
    aboutView = ft.Container(content=ft.Text("This system is a multi modal model for extracting features from video, audio and text for estimating emotion. The system has two Features, you can load a pre-recorded or run a real-time processing.",
                                               font_family='Arial',
                                               color='blue',
                                               text_align=ft.TextAlign.JUSTIFY,
                                               size=15),
                             width = 500,
                             height=200,
                             border=ft.border.all(color='blue', width=1),
                             border_radius=20,
                             alignment=ft.alignment.center)
    
    aboutLayout = ft.Row (controls=[aboutView],
                             alignment=ft.MainAxisAlignment.CENTER,
                             #horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                             visible=False)
    
    ppp = ft.Text(value="", size=15)
    uuu = ft.Text(value="", 
                  color='Blue',
                  font_family='Arial',
                  size=15)
    
    layout = ft.Container(
        content=ft.Row(
            controls=[uuu, buttonLayout, ppp, plot],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
            visible=True,
            scroll='auto'
            ), alignment=ft.alignment.center,
        padding = ft.Padding(top=20, right=0, bottom=0, left=0)
        )
    
    page.add (navbar)
    page.add (layout)
    page.add (aboutLayout)
    

audio_to_text = whisper.load_model('base')
audio_model = load_model('Audio_Emotion.h5')
text_model = load_model ('Text_Emotion.h5')
text_cv = joblib.load ('Text_Count_Vectorizer.pkl')
ps = PorterStemmer()
stopVideo = True
if __name__ == '__main__':
    ft.app(target=mainWindow, assets_dir='assets')