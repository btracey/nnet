#include <iostream>
#include <string>
#include "nnet.hpp"
using namespace std;

int main(){
	string name = "/Users/brendan/Documents/mygo/results/spalart_allmaras/su2_data/source/flatplate_sweep/trainedNet.json";
	string checkName = "/Users/brendan/Documents/mygo/results/spalart_allmaras/su2_data/source/flatplate_sweep/trainedNetpredictions.txt";
	cout << "At start"<<endl;
	int nInputs = 9;
	double* fakeData = new double[nInputs];
	for (int i = 0; i < nInputs; i++){
		fakeData[i] = double(i);
	}
	cout << "Before nnet constructor"<<endl;
	//double * fakeData[] = {1.0, 2.3, 4.9, -2.6, 1.8, 0.9, -0.98, 8.9, 4.2};
	CNeurNet mynet (name, checkName);
	cout << "After constructor"<<endl;
	double *output = new double[mynet.NumOutputs()];
	mynet.Predict(fakeData, output);
	cout << "Done predicting" <<endl;
	delete fakeData;
	cout <<"Fake data deleted"<<endl;
	delete output;
	cout <<"output deleted"<<endl;
	return 0;
}