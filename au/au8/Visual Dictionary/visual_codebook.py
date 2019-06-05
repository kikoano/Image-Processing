import os
import sys
import argparse
import cPickle as pickle
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans

def build_arg_parser():
	parser = argparse.ArgumentParser(description='Creates features for given images')
	parser.add_argument("--images", dest="input_folder", required=True, help="Folders containing the images.")
	parser.add_argument("--codebook-file", dest='codebook_file', required=True,
			help="Base file name to store the codebook")
	parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
			help="Base file name to store the feature map")

	return parser

# Loading the images from the input folder
def load_input_images(input_folder):
	input_images = []

	if not os.path.isdir(input_folder):
		raise IOError("The folder " + input_folder + " doesn't exist")

	# Parse the input folder and load the  labels
	for root, dirs, files in os.walk(input_folder):
		for filename in (x for x in files if x.endswith('.jpg')):
			input_images.append(os.path.join(root, filename))

	return input_images

class FeatureExtractor(object):
	def extract_image_features(self, img):
		# SIFT feature extractor
		kps, fvs = SIFTExtractor().compute(img)
		return fvs

	# Extract the centroids from the feature points
	def get_centroids(self, input_images, num_keypoints_to_fit=10000):
		kps_all = []

		count = 0
		for image in input_images:
			img = cv2.imread(image)
			img = resize_to_size(img, 150)
			num_dims = 128
			fvs = self.extract_image_features(img)
			kps_all.extend(fvs)
			count += len(fvs)

			if count >= num_keypoints_to_fit:
				break
			
		print "Built centroids for ", count, " keypoints"
		kmeans, centroids = Quantizer().quantize(kps_all)
		return kmeans, centroids

	def get_feature_vector(self, img, kmeans, centroids):
		return Quantizer().get_feature_vector(img, kmeans, centroids)

def extract_feature_map(input_images, kmeans, centroids):
	feature_map = []

	for image in input_images:
		temp_dict = {}
		print "Extracting features for ", image 
		img = cv2.imread(image)
		img = resize_to_size(img, 150)

		temp_dict[image] = FeatureExtractor().get_feature_vector(
					img, kmeans, centroids)

		if temp_dict[image] is not None:
			feature_map.append(temp_dict)

	return feature_map

# Vector quantization
class Quantizer(object):
	def __init__(self, num_clusters=32):
		self.num_dims = 128
		self.extractor = SIFTExtractor()
		self.num_clusters = num_clusters
		self.num_retries = 10

	def quantize(self, datapoints):
		# Create KMeans object
		kmeans = KMeans(self.num_clusters,
						n_init=max(self.num_retries, 1),
						max_iter=10, tol=1.0)

		# Run KMeans on the datapoints
		res = kmeans.fit(datapoints)

		# Extract the centroids of those clusters
		centroids = res.cluster_centers_

		return kmeans, centroids

	def normalize(self, input_data):
		sum_input = np.sum(input_data)
		if sum_input > 0:
			return input_data / sum_input
		else:
			return input_data

	# Extract feature vector from the image
	def get_feature_vector(self, img, kmeans, centroids):
		kps, fvs = self.extractor.compute(img)
		labels = kmeans.predict(fvs)
		fv = np.zeros(self.num_clusters)

		for i, item in enumerate(fvs):
			fv[labels[i]] += 1

		fv_image = np.reshape(fv, ((1, fv.shape[0])))
		return self.normalize(fv_image)

class SIFTExtractor(object):
	def compute(self, image):
		if image is None:
			print "Not a valid image"
			raise TypeError

		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		(kps, descs) = sift.detectAndCompute(gray_image, None)
		return kps, descs

# Resize the shorter dimension to 'new_size'
# while maintaining the aspect ratio
def resize_to_size(input_image, new_size=150):
	h, w = input_image.shape[0], input_image.shape[1]
	ds_factor = new_size / float(h)

	if w < h:
		ds_factor = new_size / float(w)

	new_size = (int(w * ds_factor), int(h * ds_factor))
	return cv2.resize(input_image, new_size)

if __name__=='__main__':
	args = build_arg_parser().parse_args()

	input_images = load_input_images(args.input_folder)

	# Building the codebook
	print "===== Building codebook ====="
	kmeans, centroids = FeatureExtractor().get_centroids(input_images)
	if args.codebook_file:
		with open(args.codebook_file, 'w') as f:
			pickle.dump((kmeans, centroids), f)

	# Input data and labels
	print "===== Building feature map ====="
	feature_map = extract_feature_map(input_images, kmeans, centroids)
	if args.feature_map_file:
		with open(args.feature_map_file, 'w') as f:
			pickle.dump(feature_map, f)