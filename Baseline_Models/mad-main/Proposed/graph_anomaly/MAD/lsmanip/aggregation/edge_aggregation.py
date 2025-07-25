# Get the activity timeseries of a link stream
def activity_timeseries(linkstream):
	""" Get the activity time series of a link stream
	
	:param linkstream: a linkstream object
	:type linkstream: :class: 'LinkStream'
	:return: A function t -> activity
	:rtype: dict[int, float]
	"""	
	timeseries = dict()
	weightfn = linkstream.to_weightfn()
	for triplet, value in weightfn.items():
		_,_,t = triplet
		if t not in timeseries: timeseries[t] = 0
		timeseries[t] += value
	return timeseries
