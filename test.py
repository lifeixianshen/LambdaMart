from lambdamart import LambdaMART
import numpy as np
import pandas as pd

def get_data(file_loc):
	with open(file_loc, 'r') as f:
		data = []
		for line in f:
			arr = line.split(' #')[0].split()
			score = arr[0]
			q_id = arr[1].split(':')[1]
			new_arr = [int(score), int(q_id)]
			arr = arr[2:]
			new_arr.extend(float(el.split(':')[1]) for el in arr)
			data.append(new_arr)
	return np.array(data)

def group_queries(data):
	query_indexes = {}
	for index, record in enumerate(data):
		query_indexes.setdefault(record[1], [])
		query_indexes[record[1]].append(index)
	return query_indexes


def main():
	total_ndcg = 0.0
	for i in [1,2,3,4,5]:
		print 'start Fold ' + str(i)
		training_data = get_data('Fold%d/train.txt' % (i))
		test_data = get_data('Fold%d/test.txt' % (i))
		model = LambdaMART(training_data, 300, 0.001, 'sklearn')
		model.fit()
		model.save('lambdamart_model_%d' % (i))
		# model = LambdaMART()
		# model.load('lambdamart_model.lmart')
		average_ndcg, predicted_scores = model.validate(test_data, 10)
		print average_ndcg
		total_ndcg += average_ndcg
	total_ndcg /= 5.0
	print 'Original average ndcg at 10 is: ' + str(total_ndcg)

	total_ndcg = 0.0
	for i in [1,2,3,4,5]:
		print 'start Fold ' + str(i)
		training_data = get_data('Fold%d/train.txt' % (i))
		test_data = get_data('Fold%d/test.txt' % (i))
		model = LambdaMART(training_data, 300, 0.001, 'original')
		model.fit()
		model.save('lambdamart_model_sklearn_%d' % (i))
		# model = LambdaMART()
		# model.load('lambdamart_model.lmart')
		average_ndcg, predicted_scores = model.validate(test_data, 10)
		print average_ndcg
		total_ndcg += average_ndcg
	total_ndcg /= 5.0
	print 'Sklearn average ndcg at 10 is: ' + str(total_ndcg)

	# print 'NDCG score: %f' % (average_ndcg)
	# query_indexes = group_queries(test_data)
	# index = query_indexes.keys()[0]
	# testdata = [test_data[i][0] for i in query_indexes[index]]
	# pred = [predicted_scores[i] for i in query_indexes[index]]
	# output = pd.DataFrame({"True label": testdata, "prediction": pred})
	# output = output.sort('prediction',ascending = False)
	# output.to_csv("outdemo.csv", index =False)
	# print output
	# # for i in query_indexes[index]:
	# # 	print test_data[i][0], predicted_scores[i]


if __name__ == '__main__':
	main()