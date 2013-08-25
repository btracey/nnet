package nnet

import (
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/scale"
	"github.com/gonum/floats"
	"testing"
)

const (
	netFDStep = 1e-6
	netFDTol  = 1e-8
)

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

func TestNetPredict(t *testing.T) {
	net := DefaultRegression(3, 2, 2, 4)

	is := &scale.Normal{}
	is.Mu = make([]float64, 3)
	is.Sigma = []float64{1, 1, 1}
	is.Scaled = true
	net.InputScaler = is
	os := &scale.Normal{}
	os.Mu = make([]float64, 2)
	os.Sigma = []float64{1, 1}
	os.Scaled = true
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

// Test to make sure that the predictions and derivatives match
func TestPredLossDeriv(t *testing.T) {
	net := DefaultRegression(3, 2, 2, 4)
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
	}
}
