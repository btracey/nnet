package nnet

import (
	"github.com/btracey/nnet/activator"
	"github.com/gonum/floats"
	"math"
	"testing"
)

const (
	neuronTestFDStep = 1e-6
	neuronTestFDTol  = 1e-8
)

type neuronTestAnswers struct {
	activate    float64
	combination float64
}

func neuronTest(t *testing.T, neuron Neuron, answers neuronTestAnswers, inputs, parameters []float64) {
	nParams := len(parameters)
	nInputs := len(inputs)
	// Test nParameters
	neuronSuggestedNumParameters := neuron.NumParameters(nInputs)
	if neuronSuggestedNumParameters != nParams {
		t.Errorf("Neuron.NumParameters(nInputs) returned a different number of parameters. %v passed to test, %v returned from neuron", nParams, neuronSuggestedNumParameters)
		return
	}
	// Test combination
	neuronCombination := neuron.Combine(parameters, inputs)
	if neuronCombination != answers.combination {
		t.Errorf("Neuron.Combination returned incorrect value. %v found, %v expected", neuronCombination, answers.combination)
		return
	}

	// Test activation
	neuronActivate := neuron.Activate(neuronCombination)
	if neuronActivate != answers.activate {
		t.Errorf("Neuron.Activate returned incorrect value. %v found, %v expected ", neuronActivate, neuronCombination)
		return
	}

	// Perform finite difference to check dActivateDCombination
	neuronDActivateDCombination := neuron.DActivateDCombination(neuronCombination, neuronActivate)
	fd1 := neuron.Activate(neuronCombination + neuronTestFDStep)
	fd2 := neuron.Activate(neuronCombination - neuronTestFDStep)
	fdDeriv := (fd1 - fd2) / (2 * neuronTestFDStep)
	if math.Abs(fdDeriv-neuronDActivateDCombination) > neuronTestFDTol {
		t.Errorf("Mismatch on DActivateDCombination, %v found, %v expected", neuronDActivateDCombination, fdDeriv)
		return
	}

	// Perform finite difference to check DCombineDParameters
	predDCombinationDParameters := make([]float64, len(parameters))
	neuron.DCombineDParameters(parameters, inputs, neuronCombination, predDCombinationDParameters)
	// Do finite difference to check actual derivative
	fdDCombinationDParameters := make([]float64, len(parameters))
	for i := range parameters {
		parameters[i] += neuronTestFDStep
		comb1 := neuron.Combine(parameters, inputs)
		parameters[i] -= 2 * neuronTestFDStep
		comb2 := neuron.Combine(parameters, inputs)
		parameters[i] += neuronTestFDStep
		fdDCombinationDParameters[i] = (comb1 - comb2) / (2 * neuronTestFDStep)
	}
	if !floats.EqualApprox(predDCombinationDParameters, fdDCombinationDParameters, neuronTestFDTol) {
		t.Errorf("DCombinationDParameters mismatch. %v found, %v expected", predDCombinationDParameters, fdDCombinationDParameters)
		return
	}

	// Perform finite difference to check DCombineDInput
	predDCombinationDInput := make([]float64, len(inputs))
	neuron.DCombineDInput(parameters, inputs, neuronCombination, predDCombinationDInput)

	// Do finite difference to check actual derivative
	fdDCombinationDInput := make([]float64, len(inputs))
	for i := range inputs {
		inputs[i] += neuronTestFDStep
		comb1 := neuron.Combine(parameters, inputs)
		inputs[i] -= 2 * neuronTestFDStep
		comb2 := neuron.Combine(parameters, inputs)
		inputs[i] += neuronTestFDStep
		fdDCombinationDInput[i] = (comb1 - comb2) / (2 * neuronTestFDStep)
	}
	if !floats.EqualApprox(predDCombinationDInput, fdDCombinationDInput, neuronTestFDTol) {
		t.Errorf("DCombinationDInputs mismatch. %v found, %v expected", predDCombinationDInput, fdDCombinationDInput)
		return
	}

}

func TestSumNeuron(t *testing.T) {
	tanhLinear := &SumNeuron{activator.LinearTanh{}}
	inputs := []float64{1, 2, 3, 4}
	params := []float64{0.1, -1.1, 2.4, -8, 0.2}
	trueCombination := 0.1 - 2.2 + 7.2 - 32 + 0.2
	answers := neuronTestAnswers{
		combination: trueCombination,
		activate:    tanhLinear.Activate(trueCombination),
		//dActivateDCombine: tanhLinear.DActivateDCombination(sum, output),
	}
	neuronTest(t, tanhLinear, answers, inputs, params)
}
