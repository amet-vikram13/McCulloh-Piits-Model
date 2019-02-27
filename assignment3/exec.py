from find import *

def main() :
	df = pd.read_csv("heart.csv")
	model = DeepNeuralNetwork([13,45,45,45,45,1])
	model.process_data(df)
	maxx = {"accu":0,"mse":0,"plt_cost":0,"plt_mse":0,"plt_accu":0}	
	for round in range(200) :
		model.train_test_split()
		p = model.initialize_parameters()
		t = model.run(100,0.001,False)
		if t["accu"] > maxx["accu"] :
			     maxx = t
		print("round {} done!".format(round))	
	return maxx



