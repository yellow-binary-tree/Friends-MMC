
import os
import sys
import csv
import glob
import json
import math
import shutil
import pickle
import random
import logging
import argparse
import subprocess
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import torch
# from moviepy.editor import VideoFileClip

sys.path.append('./TalkNet_ASD')
from TalkNet_ASD.talkNet import talkNet


def active_speaker_detection(args):
    import python_speech_features
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    s.eval()
    durationSet = {1,2,4,6}     # To make the result more reliable
    for episode_name in tqdm(sorted(os.listdir(args.face_track_path))[args.start_idx:args.end_idx]):
        if os.path.exists(os.path.join(args.output_path, '%s.pkl' % episode_name[:6])):
            all_scores = pickle.load(open(os.path.join(args.output_path, '%s.pkl' % episode_name[:6]), 'rb'))
        else:
            all_scores = dict()
        # all_scores = dict()     # now we rerun all data
        if not os.path.isdir(os.path.join(args.face_track_path, episode_name)): continue
        print('processing: %s' % episode_name)
        for utterance_name in os.listdir(os.path.join(args.face_track_path, episode_name)):
            files = glob.glob("%s/*.avi" % os.path.join(args.face_track_path, episode_name, utterance_name))
            files.sort()
            if utterance_name in all_scores and len(all_scores[utterance_name]) == len(files):
                continue        # this utterance has already been successfully detected before
            face_scores = []
            for face_i, file in enumerate(files):
                sample_rate, audio = wavfile.read(file.replace(".avi", ".wav"))
                audioFeature = python_speech_features.mfcc(audio, sample_rate, numcep = 13, winlen = 0.025, winstep = 0.010, nfft=51200)
                video = cv2.VideoCapture(file)
                videoFeature = []
                while video.isOpened():
                    ret, frames = video.read()
                    if ret == True:
                        face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                        face = cv2.resize(face, (224,224))
                        face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                        videoFeature.append(face)
                    else:
                        break
                video.release()
                try:        # some very short videos may raise error. but as they are short so they are not likely to be speakers
                    videoFeature = np.array(videoFeature)
                    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
                    audioFeature = audioFeature[:int(round(length * 100)),:]
                    videoFeature = videoFeature[:int(round(length * 25)),:,:]
                except:
                    logging.warning('error encountered when loading video feature of %s' % file)
                    face_scores.append(list())
                    continue

                # clip the longer one of audioFeature or videoFeature so that they will match, or may cause error when performing asd
                max_length = min(videoFeature.shape[0], audioFeature.shape[0] // 4)
                videoFeature, audioFeature = videoFeature[:max_length], audioFeature[:max_length * 4]
                allScore = []   # Evaluation use TalkNet
                for duration in durationSet:
                    try:
                        batchSize = int(math.ceil(length / duration))
                        scores = []
                        with torch.no_grad():
                            for i in range(batchSize):
                                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                                embedA = s.model.forward_audio_frontend(inputA)
                                embedV = s.model.forward_visual_frontend(inputV)	
                                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                                out = s.model.forward_audio_visual_backend(embedA, embedV)
                                score = s.lossAV.forward(out, labels = None)
                                scores.extend(score)
                        allScore.append(scores)
                    except:
                        logging.warning('error encountered when performing asd %s %d' % (file, duration))

                allScore = np.mean(np.array(allScore), axis = 0).tolist()
                # all_scores[utterance_name][face_i] = allScore
                face_scores.append(allScore)
            all_scores[utterance_name] = face_scores

        with open(os.path.join(args.output_path, '%s.pkl' % episode_name[:6]), 'wb') as f_out:
            pickle.dump(all_scores, f_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TalkNet Demo")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100000000)

    parser.add_argument('--face_track_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--pretrainModel',         type=str, default="./TalkNet_ASD/pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    if args.start_idx > len(os.listdir(args.face_track_path)):
        sys.exit()
    if args.end_idx > len(os.listdir(args.face_track_path)):
        args.end_idx = len(os.listdir(args.face_track_path))
    active_speaker_detection(args)
