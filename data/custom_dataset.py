import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

def _parse_labels(fn):
    d = {}
    for row in open(fn,'r'):
        row = row.strip().split(',')
        f,tag = row[0],int(row[1])
        d[f] = tag
    return d

class CustomDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase+'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase+'B')
        self.dir_Asup = os.path.join(opt.dataroot, opt.phase+'Asup')
        self.dir_Bsup = os.path.join(opt.dataroot, opt.phase+'Bsup')

        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)
        self.Asup_paths = make_dataset(self.dir_Asup, opt.max_dataset_size)
        self.Bsup_paths = make_dataset(self.dir_Bsup, opt.max_dataset_size)
        self.Asup_labels = _parse_labels(self.dir_Asup+'_labels.txt')
        self.Bsup_labels = _parse_labels(self.dir_Bsup+'_labels.txt')

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.Asup_size = len(self.Asup_paths)
        self.Bsup_size = len(self.Bsup_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, idx):

        A_path = self.A_paths[idx % self.A_size]
        if self.opt.serial_batches:
            index_B = idx % self.B_size
            index_Asup = idx % self.Asup_size
            index_Bsup = idx % self.Bsup_size
        else:
            index_B = random.randint(0, self.B_size-1)
            index_Asup = random.randint(0, self.Asup_size-1)
            index_Bsup = random.randint(0, self.Bsup_size-1)
        B_path,Asup_path,Bsup_path = self.B_paths[index_B],self.Asup_paths[index_Asup],self.Bsup_paths[index_Bsup]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        Asup_img = Image.open(Asup_path).convert('RGB')
        Bsup_img = Image.open(Bsup_path).convert('RGB')
        Asup_label = self.Asup_labels[Asup_path.split('/')[-1]]
        Bsup_label = self.Bsup_labels[Bsup_path.split('/')[-1]]

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        A_sup = self.transform_A(Asup_img)
        B_sup = self.transform_B(Bsup_img)

        item = {
            'A': A,
            'B': B,
            'A_sup': A_sup,
            'B_sup': B_sup,
            'A_sup_label': Asup_label,
            'B_sup_label': Bsup_label,
            'A_paths': A_path,
            'B_paths': B_path,
            'A_sup_paths': Asup_path,
            'B_sup_paths': Bsup_path
        }

        return item

    def __len__(self):
        return max(max(self.A_size,self.B_size),max(self.Asup_size,self.Bsup_size))

