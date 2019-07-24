import numpy as np 
import cv2
import glob 
import random
import os
import errno
import sys
import math
from PIL import Image


def Distance(p1,p2):
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
	if (scale is None) and (center is None):
		return image.rotate(angle=angle, resample=resample)
	nx,ny = x,y = center
	sx=sy=1.0
	if new_center:
		(nx,ny) = new_center
	if scale:
		(sx,sy) = (scale, scale)
	cosine = math.cos(angle)
	sine = math.sin(angle)
	a = cosine/sx
	b = sine/sx
	c = x-nx*a-ny*b
	d = -sine/sy
	e = cosine/sy
	f = y-nx*d-ny*e
	return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def cropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.3,0.3), dest_sz = (128,128)):
  # calculate offsets in original image
	offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
	offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
	# get the direction
	eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
	# calc rotation angle in radians
	rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
	# distance between them
	dist = Distance(eye_left, eye_right)
	# calculate the reference eye-width
	reference = dest_sz[0] - 2.0*offset_h
	# scale factor
	scale = float(dist)/float(reference)
	# rotate original around the left eye
	image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
	# crop the rotated image
	crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
	crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
	image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
	# resize it
	image = image.resize(dest_sz, Image.ANTIALIAS)
	return image


def preprocess():
	try:
		os.makedirs("./data/TSKinFace_Data/TSKinFace_Cropped")
		os.makedirs("./data/TSKinFace_Data/TSKinFace_Cropped/FMD")
		os.makedirs("./data/TSKinFace_Data/TSKinFace_Cropped/FMS")
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
	
	srcPath = "./data/TSKinFace_Data/TSKinFace_source"
	relationType = ["FMS", "FMD"]
	for relation in relationType:
		filePath = srcPath + "/" + relation + "_information/" + relation + "_Eye_combine.txt"
		with open(filePath) as fp:
			for line in fp:
				line = line.strip()
				if not line:
					continue
				i = line.split(' ')
				imgName = i[0].split('\\')[1]
				role = i[1]
				imgPath = srcPath + "/" + relation + "/" + imgName
				image = Image.open(imgPath)
				faceAligned = cropFace(image, (float(i[2]), float(i[3])), (float(i[4]), float(i[5])) )
				newImgName = imgName.split(".")[0] + "-" + role + ".jpg"
				newPath = "./data/TSKinFace_Data/TSKinFace_Cropped/" + relation + "/" + newImgName
				faceAligned.save(newPath, "JPEG")

	cnt = 1
	with open("./data/TSKinFace_Data/TSKinFace_source/FMSD_information/FMSD_Eye_combine.txt") as fp:
		for line in fp:
			line = line.strip()
			if not line:
				cnt += 1
				continue
			i = line.split(' ')
			imgName = i[0].split('\\')[1]
			role = i[1]
			imgPath = "./data/TSKinFace_Data/TSKinFace_source/FMSD/" + imgName
			image = Image.open(imgPath)
			faceAligned = cropFace(image, (float(i[2]), float(i[3])), (float(i[4]), float(i[5])))
			FMS_idx = 285 + cnt
			FMD_idx = 274 + cnt
			newFMDpath = "./data/TSKinFace_Data/TSKinFace_Cropped/FMD/FMD-" + str(FMD_idx) + "-" + role + ".jpg"
			newFMSpath = "./data/TSKinFace_Data/TSKinFace_Cropped/FMS/FMS-" + str(FMS_idx) + "-" + role + ".jpg" 
			if role == 'M' or role == 'F':	
				faceAligned.save(newFMDpath, "JPEG")
				faceAligned.save(newFMSpath, "JPEG")
			elif role == 'D':
				faceAligned.save(newFMDpath, "JPEG")
			else:
				faceAligned.save(newFMSpath, "JPEG")
			

def custom_rand(relType, curr_idx):
	if relType == 'FMD':
		nFMD = len(glob.glob("./data/TSKinFace_Data/TSKinFace_Cropped/FMD/*.jpg"))/3 
		randIdx = random.randint(1, int(nFMD))  
		return custom_rand(relType, curr_idx) if randIdx == curr_idx else randIdx
	else:
		nFMS = len(glob.glob("./data/TSKinFace_Data/TSKinFace_Cropped/FMS/*.jpg"))/3 
		randIdx = random.randint(1, int(nFMS))
		return custom_rand(relType, curr_idx) if randIdx == curr_idx else randIdx

class Data_reader():
	def __init__(self):
		print("loading data now")
		self.load_data()
		random.shuffle(self.data)
		self.trainPos = 0
		self.testPos = 0
		self.trainData = []
		self.testData = []

        def process_image(self, img):
            return np.float32(img) / 127.5 - 1.

	def set_traintest_split(self,split = 0.7):
		total_dataSize = len(self.data)
		self.trainData = self.data[0: int(math.floor(total_dataSize * split))]
		self.testData = self.data[int(math.ceil(total_dataSize * split)): total_dataSize]

	def load_data(self):
		print('Loading data...')
		data = []
		
		relationType = ["FMS", "FMD"]
		for relation in relationType:
			nTriSubject = len(glob.glob("./data/TSKinFace_Data/TSKinFace_Cropped/" + relation + "/*.jpg"))/3
			upperBound = int(nTriSubject) + 1
			for j in range(1, upperBound):
				mother = []
				father = []
				mother_likedness = random.random()
				pos_child = []
				neg_children =[]
				for i in glob.glob("./data/TSKinFace_Data/TSKinFace_Cropped/" + relation + "/" + relation + "-" + str(j) + "-*.jpg"):
					img = cv2.cv2.imread(i)
					img = cv2.cv2.resize(img, (128, 128))
                                        img = self.process_image(img)
					i = i.split('/')[-1]
					i = i.split('.')[0]
					i = i.split('-')
					curr_idx = i[1]
					role = i[2]
					if role == 'M':
						mother = img
					elif role == 'F':
						father = img
					else:
						pos_child = img
				rand_relType = np.random.randint(0,2,5)
				for relType in rand_relType:
					rel = relationType[relType]
					rand_idx = custom_rand(rel, curr_idx)
					child_role = 'D' if rel=='FMD' else 'S'
					imgPath = './data/TSKinFace_Data/TSKinFace_Cropped/' + rel + '/' + rel + "-" + str(rand_idx) + "-" + child_role + ".jpg"
					neg_child_img = cv2.cv2.imread(imgPath)
					neg_child_img = cv2.cv2.resize(neg_child_img, (128, 128))
                                        neg_child_img = self.process_image(neg_child_img)

					neg_children = neg_child_img
                                        break
				data.append([father, mother, mother_likedness, pos_child, neg_children])
			self.data = data
			print('Load finished.')

	def get_next_train_batch(self, bsize):
		if self.trainPos + bsize > len(self.trainData):
			random.shuffle(self.trainData)
			self.trainPos = 0

		batch = self.trainData[self.trainPos: self.trainPos+bsize]
		self.trainPos += bsize
		father_batch, mother_batch, mother_likedness, pos_child_batch, neg_children_batch = list(zip(*batch))
		return father_batch, mother_batch, mother_likedness, pos_child_batch, neg_children_batch



	def get_next_test_batch(self,bsize):
		if self.trainPos + bsize > len(self.testData):
			random.shuffle(self.testData)
			self.testPos = 0

		batch = self.testData[self.testPos: self.testPos+bsize]
		self.testPos += bsize
		
		father_batch, mother_batch, mother_likedness, pos_child_batch,_ = list(zip(*batch))
		return father_batch, mother_batch, mother_likedness, pos_child_batch

'''
def main():
	data_loader = Data_reader()
	data_loader.set_traintest_split(0.7)
	father_batch, mother_batch, mother_likedness, pos_child_batch, neg_children_batch = data_loader.get_next_train_batch(2)
	for father in father_batch:
		cv2.cv2.imshow('father', father)
		cv2.cv2.waitKey(0)
	for mother in mother_batch:
		cv2.cv2.imshow('mother', mother)
		cv2.cv2.waitKey(0)
	for child in pos_child_batch:
		cv2.cv2.imshow('child', child)
		cv2.cv2.waitKey(0)
	for neg_children in neg_children_batch:
		for neg_c in neg_children:
			cv2.cv2.imshow('neg_child', neg_c)
			cv2.cv2.waitKey(0)
	cv2.cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
'''
