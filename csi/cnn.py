import os
import json
import argparse

from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
device = torch.device('cuda:0')
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training


def calculate_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class FaceDataset:
    def __init__(self, base_folder, transform='default', split='train', debug=False):
        self.frames_path = os.path.join(base_folder, 'images')
        self.debug = debug
        self.split = split

        if split == 'test':
            metadata = json.load(open(os.path.join(base_folder, 'test-metadata.json')))
        elif split == 'test-noisy':
            metadata = json.load(open(os.path.join(base_folder, 'test-noisy-metadata.json')))
        else:
            metadata = json.load(open(os.path.join(base_folder, 'train-metadata.json')))

        self.examples, self.labels = list(), list()
        frames_met = set()
        for dialog_data in metadata:
            for frame_data in dialog_data:
                # use season 01 as valid set
                if split == 'valid' and not frame_data['frame'].startswith('s01'):
                    continue
                if split == 'train' and frame_data['frame'].startswith('s01'):
                    continue
                if frame_data['frame'] in frames_met:
                    continue
                frames_met.add(frame_data['frame'])

                faces = frame_data['faces']      # [(bbox, id), (bbox, id), ...]

                for bbox, face_label in faces:
                    bbox = [max(i, 0) for i in bbox]
                    bbox = [min(bbox[0], 1280), min(bbox[1], 720), min(bbox[2], 1280), min(bbox[3], 720)]
                    if calculate_area(bbox) <= 0:
                        continue
                    label = int(face_label == frame_data['speaker'])
                    self.examples.append((frame_data['frame'], face_label, bbox, label))
                    self.labels.append(label)

        if transform == 'default':
            self.train_transform = transforms.Compose([
                np.float32,
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=20, shear=20),
                transforms.Resize((512, 512)),
                fixed_image_standardization
            ])
            self.valid_transform = transforms.Compose([
                np.float32,
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                fixed_image_standardization
            ])
        else:
            self.train_transform = transform
            self.valid_transform = transform

        print('loaded %d examples' % len(self))
        print('example data:', [i.size() if isinstance(i, torch.FloatTensor) else i for i in self[0]])

    def __len__(self):
        return len(self.examples) if not self.debug else 32

    def __getitem__(self, index):
        frame_name, face_label, bbox, label = self.examples[index]
        image = cv2.imread(os.path.join(self.frames_path, frame_name + '.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # if self.split in ['valid', 'test']:
        if self.split == 'valid' or self.split.startswith('test'):
            image = self.valid_transform(image)
            return image, label, frame_name, face_label
        else:
            image = self.train_transform(image)
            return image, label



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='train-friends')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_pretrained', type=int, default=1)

    parser.add_argument('--data_base_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--decay_interval', type=int, default=3)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_ckpt', type=str)
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_path, exist_ok=True)

    if args.func.split('-')[0] == 'train':
        if args.load_pretrained:
            resnet = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=2).to(device)
        else:
            resnet = InceptionResnetV1(classify=True, num_classes=2).to(device)
        softmax = torch.nn.Softmax()

        train_dataset = FaceDataset(args.data_base_folder, split='train', debug=args.debug)
        valid_dataset = FaceDataset(args.data_base_folder, split='valid', debug=args.debug)
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }

        writer = SummaryWriter(os.path.join(args.output_path, 'tensorboard_logs'))
        writer.iteration, writer.interval = 0, 10
        optimizer = optim.Adam(resnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, list(range(args.decay_interval, args.num_epochs, args.decay_interval)))

        best_acc = 0
        for epoch_i in range(args.num_epochs):
            print('\nEpoch {}/{}'.format(epoch_i + 1, args.num_epochs))
            resnet.train()
            train_loss, train_metrics = training.pass_epoch(
                resnet, loss_fn, train_loader, optimizer, scheduler,
                batch_metrics=metrics, show_running=False, device=device,
                writer=writer
            )

            with torch.no_grad():
                resnet.eval()
                test_output = dict()
                for images, labels, frame_names, face_labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    y_preds = softmax(resnet(images))[:, 1]
                    for y_pred, frame_name, face_label, label in zip(y_preds, frame_names, face_labels, labels):
                        y_pred = float(y_pred.cpu())
                        label = int(label.cpu())
                        if frame_name not in test_output:
                            test_output[frame_name] = dict()
                        test_output[frame_name][face_label] = {'pred': y_pred, 'label': label}

            val_metrics = dict()
            frame_acc_dict, dialog_acc_list = dict(), list()
            for frame_id, frame_pred in test_output.items():
                y_preds = [i['pred'] for i in frame_pred.values()]
                y_golds = [i['label'] for i in frame_pred.values()]
                if not any(y_golds):        # all 0, speaker not in frame
                    frame_acc_dict[frame_id] = 0
                else:
                    frame_acc_dict[frame_id] = (np.argmax(y_preds) == np.argmax(y_golds))

            val_metrics['frame_acc'] = sum(frame_acc_dict.values()) / len(frame_acc_dict)
            val_metadata = json.load(open(os.path.join(args.data_base_folder, 'train-metadata.json')))
            for dialog_data in val_metadata:
                for frame_data in dialog_data:
                    if frame_data['frame'] in frame_acc_dict:
                        dialog_acc_list.append(frame_acc_dict[frame_data['frame']])
            val_metrics['dialog_acc'] = sum(dialog_acc_list) / len(dialog_acc_list)

            for key in val_metrics:
                writer.add_scalar('valid/' + key, val_metrics[key], epoch_i)

            print('epoch %d' % epoch_i)
            print('dialog level acc: {}'.format(val_metrics['dialog_acc']))
            print('val acc: {}, best_acc: {}'.format(val_metrics['frame_acc'], best_acc))
            if val_metrics['frame_acc'] > best_acc:
                best_acc = val_metrics['frame_acc']
                torch.save(resnet.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
                json.dump(test_output, open(os.path.join(args.output_path, 'valid_output.json'), 'w'))
        writer.close()

    elif args.func.startswith('test'):
        resnet = InceptionResnetV1(classify=True, num_classes=2).to(device)
        resnet.eval()
        print('loading checkpoint from %s' % args.model_ckpt)
        resnet.load_state_dict(torch.load(args.model_ckpt))
        softmax = torch.nn.Softmax()

        for split in ['test', 'test-noisy']:
            test_dataset = FaceDataset(args.data_base_folder, split=split, debug=args.debug)
            test_loader =  DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

            with torch.no_grad():
                resnet.eval()
                test_output = dict()
                for images, labels, frame_names, face_labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    y_preds = softmax(resnet(images))[:, 1]
                    for y_pred, frame_name, face_label, label in zip(y_preds, frame_names, face_labels, labels):
                        y_pred = float(y_pred.cpu())
                        label = int(label.cpu())
                        if frame_name not in test_output:
                            test_output[frame_name] = dict()
                        test_output[frame_name][face_label] = {'pred': y_pred, 'label': label}

            # evalate the results
            test_metrics = dict()
            frame_acc_dict, dialog_acc_list = dict(), list()
            for frame_id, frame_pred in test_output.items():
                y_preds = [i['pred'] for i in frame_pred.values()]
                y_golds = [i['label'] for i in frame_pred.values()]
                if not any(y_golds):        # all 0, speaker not in frame
                    frame_acc_dict[frame_id] = 0
                else:
                    frame_acc_dict[frame_id] = (np.argmax(y_preds) == np.argmax(y_golds))

            test_metrics['frame_acc'] = sum(frame_acc_dict.values()) / len(frame_acc_dict)
            if split.startswith('test'):
                test_metadata = json.load(open(os.path.join(args.data_base_folder, f'{split}-metadata.json')))
            else:
                test_metadata = [example for example in json.load(open(os.path.join(args.data_base_folder, 'train-metadata.json'))) if example[0]['frame'].startswith('s01')]

            for dialog_data in test_metadata:
                for frame_data in dialog_data:
                    if frame_data['frame'] in frame_acc_dict:
                        dialog_acc_list.append(frame_acc_dict[frame_data['frame']])
                    else:
                        dialog_acc_list.append(0)       # no faces in frame
            test_metrics['dialog_acc'] = sum(dialog_acc_list) / len(dialog_acc_list)

            print('dialog level acc: {}'.format(test_metrics['dialog_acc']))
            print('frame level acc: {}'.format(test_metrics['frame_acc']))
            json.dump(test_output, open(os.path.join(args.output_path, '%s_output.json' % split), 'w'))