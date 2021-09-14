import sys
from ..BaseData_class import BaseData_class
import scipy as sc
import os
import numpy as np

class Data_class(BaseData_class):
	"""
	Class allowing to deal with the importation of the precomputed fingerprints.
	"""
	def __init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, nb_files, path_files, CRBrequired = False):
		"""
		New Data_class.
		"""
		BaseData_class.__init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, CRBrequired=CRBrequired)
		self.nb_files = nb_files		
		self.path_files = path_files
		
	def load_urls(self):
		"""
		Transform the text file containing the urls associated to the precomputed fingerprints.
		"""
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path, '../../../CRB/MRF/Offline/loading_data/'+self.urls_file)
		with open(path, encoding="ISO-8859-1") as f:
			for line in f:
				urls = line.strip().split(',')
		return(urls)
		
	def load_data_from_web(self,num):
		"""
		Load the file number 'num' containing precomputed fingerprints from the web.
		"""
		cmd = "wget --quiet -O data.mat "+self.urls[num]
		run(cmd,shell=True)
		data = sc.io.loadmat('data.mat')
		return data['s'].T, np.concatenate((data['m0s'],data['T1'],data['T2f'],data['R'],data['T1'],data['T2s']), axis=1)

	def load_data(self,num):
		"""
		Load the file number 'num' containing precomputed fingerprints previously saved in the folder 'loading_data'.
		"""
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path,  'loading_data/') #os.path.join(my_path,  '../../../CRB/MRF/Offline/loading_data/')
		path = os.path.join(path, self.path_files)
		try:
			data = np.load(path+'/fingerprints'+str(num)+'.npy')
			params = np.load(path+'/params'+str(num)+'.npy')
			if self.CRBrequired:
				CRBs = np.load(path+'/CRBs'+str(num)+'.npy')
			else:
				CRBs = None
		except:
			print("Error occured trying to load the file number "+str(num))
			raise
		return data, params, CRBs
		
	def load_CRBs(self,num):
		"""
		Load the file number 'num' containing precomputed fingerprints previously saved in the folder 'loading_data'.
		"""
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path, 'loading_data') 
		try:
			data = np.load(path+'/CRBs'+str(num)+'.npy')
		except:
			print("Error occured trying to load the file number "+str(num))
			raise
		return data
		

	def sample(self):
		"""
		Define the sampling strategy used to built the precomputed fingerprints files. This method is only informative and will not be used in this offline framework.
		"""
		random.seed()
		np.random.seed()
		m0s = random.uniform(0,0.7)
		t1 = 2.8 * random.random() + 0.2
		t2f = t1 * ( random.random() * 0.5 + 0.005 )
		r = 490 * random.random() + 10
		t2s= 0.2 * 10**(-3) + random.random() * 150 * 10**(-3)
		return(np.array([m0s,t1,t2,r,t2s]))