import csv
from .. import LinkStream

def from_csv(path, weighted=False):
	""" Read a link stream from a csv file 

	:param path: Absolute path of CSV file
	:param weighted: 'True' if triplets are weighted
	:type path: string
	:type weighted: Boolean, defaults to 'False'
	:type linkstream: class:'LinkStream', defaults to None
	:return: A link stream object of class class:'LinkStream'
	"""
	# Create link stream 
	linkstream = LinkStream()

	# Read csv file
	with open(path, 'r') as file: 
		reader = csv.reader(file, delimiter=',')
		for line in reader:
			# Add triplet (u, v,t)
			triplet = (line[0], line[1], int(line[2]))
			linkstream.add_triplet(triplet)
			# If weighted, add it to triplets
			if weighted: linkstream.set_triplet_weight(triplet, float(line[3]))

	linkstream.isweighted = weighted
	linkstream.set_stats()
	return linkstream
