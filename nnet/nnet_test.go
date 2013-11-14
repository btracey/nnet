package nnet

import (
	"encoding/json"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/scale"
	"github.com/gonum/floats"
	"math/rand"
	"testing"

	"fmt"
	"reflect"
)

const (
	netFDStep = 1e-7
	netFDTol  = 2e-8
)

func TestJSON(t *testing.T) {
	net := DefaultRegression(3, 4, 1, 10)

	nInputs := 3
	nOutputs := 4
	rInput := RandomData(nInputs, 100)
	rOutput := RandomData(nOutputs, 100)
	net.InputScaler = &scale.Linear{}
	net.InputScaler.SetScale(rInput)
	net.OutputScaler = &scale.Normal{}
	net.OutputScaler.SetScale(rOutput)
	net.Losser = loss.LogSquared{}

	data, err := json.Marshal(net)
	if err != nil {
		t.Errorf("Error marshaling net: " + err.Error())
	}
	net2 := &Net{}
	err = net2.UnmarshalJSON(data)
	if err != nil {
		t.Errorf("Error unmarshaling: " + err.Error())
	}
	if !reflect.DeepEqual(net, net2) {
		t.Errorf("Net not equal after encoding and decoding")
	}
}

func TestGob(t *testing.T) {
	nInputs := 4
	nOutputs := 5
	nLayers := 1
	nNeuronsPerLayer := 7
	rInput := RandomData(nInputs, 100)
	rOutput := RandomData(nOutputs, 100)

	net := DefaultRegression(nInputs, nOutputs, nLayers, nNeuronsPerLayer)
	net.InputScaler = &scale.Linear{}
	net.InputScaler.SetScale(rInput)
	net.OutputScaler = &scale.Normal{}
	net.OutputScaler.SetScale(rOutput)
	net.Losser = loss.LogSquared{}

	bytes, err := net.GobEncode()
	if err != nil {
		t.Error(err)
		return
	}

	predictionsNet1, err := net.PredictSlice(rInput)
	if err != nil {
		t.Error("Error predicting on net 1")
	}

	net2 := &Net{}
	err = net2.GobDecode(bytes)
	if err != nil {
		t.Errorf("Error decoding net2: %v ", err)
		return
	}

	if !reflect.DeepEqual(net, net2) {
		t.Errorf("Nets don't match after encoding and decoding")
	}

	predictionsNet2, err := net2.PredictSlice(rInput)
	if err != nil {
		t.Error("Error predicting on net 2")
	}

	for i := range predictionsNet1 {
		if !floats.EqualApprox(predictionsNet1[i], predictionsNet2[i], 1e-14) {
			t.Errorf("Predictions don't match")
			return
		}
	}

	filename := "testnet.gob"
	err = net.Save(filename)
	if err != nil {
		t.Errorf("Error saving net: %v", err)
	}

	net3, err := Load(filename)
	if err != nil {
		t.Errorf("Error loading net: %v", err)
	}
	if !reflect.DeepEqual(net, net3) {
		t.Errorf("Not equal after save and load")
	}

	net4, err := Load(filename)
	if err != nil {
		t.Errorf("Error loading net: %v", err)
	}
	if !reflect.DeepEqual(net, net4) {
		t.Errorf("Not equal after save and load")
	}
}

func RandomData(size int, numberOfSamples int) [][]float64 {
	data := make([][]float64, numberOfSamples)
	for i := range data {
		data[i] = make([]float64, size)
		for j := range data[i] {
			data[i][j] = rand.Float64()
		}
	}
	return data
}

func TestProcessNeuron(t *testing.T) {
	neuron := &TanhNeuron
	parameters := []float64{1, 2, 3, 4}
	inputs := []float64{0.4, 0.6, 0.8}
	combination, output := ProcessNeuron(neuron, parameters, inputs)
	trueComb := neuron.Combine(parameters, inputs)
	trueOut := neuron.Activate(trueComb)
	if combination != trueComb {
		t.Errorf("Combination doesn't match. %v found, %v expected", combination, trueComb)
	}
	if output != trueOut {
		t.Errorf("Combination doesn't match. %v found, %v expected", output, trueOut)
	}
}

func TestNetPredictMatchPredict(t *testing.T) {
	net := DefaultRegression(3, 2, 2, 4)

	is := &scale.Normal{}
	is.Mu = make([]float64, 3)
	is.Sigma = []float64{1, 1, 1}
	is.Scaled = true
	is.Dim = 3
	net.InputScaler = is
	os := &scale.Normal{}
	os.Mu = make([]float64, 2)
	os.Sigma = []float64{1, 1}
	os.Scaled = true
	is.Dim = 2
	net.OutputScaler = os
	input := []float64{1, 2, 3}
	net.RandomizeParameters()

	//predOutput1 := make([]float64, net.Outputs())
	predOutput2 := make([]float64, net.Outputs())

	predictTmpMemory := net.NewPredictTmpMemory()

	predOutput1, err := net.Predict(input)
	if err != nil {
		t.Errorf("Error predicting")
	}

	Predict(input, net, predOutput2, predictTmpMemory.combinations, predictTmpMemory.outputs)

	if !floats.EqualApprox(predOutput1, predOutput2, 1e-15) {
		t.Errorf("net.Predict and Predict don't match")
	}
}

func TestPredictSliceMatchPredict(t *testing.T) {
	nInputs := 3
	net := DefaultRegression(nInputs, 2, 2, 4)
	is := &scale.Normal{}
	is.Mu = make([]float64, 3)
	is.Sigma = []float64{1, 1, 1}
	is.Scaled = true
	is.Dim = 3
	net.InputScaler = is
	os := &scale.Normal{}
	os.Mu = make([]float64, 2)
	os.Sigma = []float64{1, 1}
	os.Scaled = true
	os.Dim = 2
	net.OutputScaler = os
	//input := []float64{1, 2, 3}
	net.RandomizeParameters()

	nSamples := 10
	data := RandomData(nInputs, nSamples)
	data1 := make([]float64, nInputs)
	for i := range data[0] {
		data1[i] = data[0][i]
	}

	var err error
	predictions1 := make([][]float64, nSamples)
	for i := range data {
		predictions1[i], err = net.Predict(data[i])
		if err != nil {
			t.Errorf("Error using net.Predict: %v", err)
			return
		}
	}
	// Check that data matches
	if !floats.EqualApprox(data[0], data1, 1e-15) {
		t.Errorf("Data doesn't match after predict")
		return
	}

	predictions2, err := net.PredictSlice(data)
	if err != nil {
		t.Errorf("Error predicting in PredictSlice: %v", err)
		return
	}
	if !floats.EqualApprox(data[0], data1, 1e-15) {
		t.Errorf("Data doesn't match after PredictSlice")
		return
	}

	for i := range predictions1 {
		if !floats.EqualApprox(predictions1[i], predictions2[i], 1e-14) {
			t.Errorf("Prediction %v doesn't match", i)
			fmt.Println(predictions1, "\n", predictions2)
			return
		}
	}

}

// Test to make sure that the predictions and derivatives match
func TestPredLossDeriv(t *testing.T) {
	net := DefaultRegression(3, 2, 2, 50)
	net.Losser = &loss.SquaredDistance{}
	input := []float64{1, 2, 3}
	truth := []float64{1.2, 2.2}
	net.RandomizeParameters()
	tmp := net.NewPredLossDerivTmpMemory()

	prediction := make([]float64, len(truth))
	dLossDParam, dLossFlat := net.NewPerParameterMemory()
	_, FDdLossFlat := net.NewPerParameterMemory()
	params := make([]float64, net.TotalNumParameters())
	net.ParametersSlice(params)
	PredLossDeriv(input, truth, net, tmp, prediction, dLossDParam)

	predictTmp := make([]float64, len(truth))
	dLossDPredTmp := make([]float64, len(truth))
	combinations := net.NewPerNeuronMemory()
	outputs := net.NewPerNeuronMemory()
	// Compare finite difference
	for i := range params {
		params[i] += netFDStep
		net.SetParametersSlice(params)
		Predict(input, net, predictTmp, combinations, outputs)
		loss1 := net.Losser.LossAndDeriv(predictTmp, truth, dLossDPredTmp)
		params[i] -= 2 * netFDStep
		net.SetParametersSlice(params)
		Predict(input, net, predictTmp, combinations, outputs)
		loss2 := net.Losser.LossAndDeriv(predictTmp, truth, dLossDPredTmp)
		params[i] += netFDStep
		FDdLossFlat[i] = (loss1 - loss2) / (2 * netFDStep)
	}
	if !floats.EqualApprox(dLossFlat, FDdLossFlat, netFDTol) {
		t.Errorf("Finite difference doesn't match derivative")
		for i := range dLossFlat {
			fmt.Println(i, dLossFlat[i], FDdLossFlat[i], dLossFlat[i]-FDdLossFlat[i])
		}
	}
}
