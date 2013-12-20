package nnet

import (
	"github.com/gonum/floats"
)

// ProcessNeuron computes
func ProcessNeuron(neuron Neuron, parameters, inputs []float64) (combination, output float64) {
	combination = neuron.Combine(parameters, inputs)
	output = neuron.Activate(combination)
	return
}

func DLossNeuron(n Neuron, parameters []float64, inputs []float64, combination, output, dLossDOutput float64, dLossDParam, dLossDInput []float64) {
	dOutputDCombination := n.DActivateDCombination(combination, output)
	dLossDCombination := dLossDOutput * dOutputDCombination

	// Store DCombineDParameters into dLossDParam
	n.DCombineDParameters(parameters, inputs, combination, dLossDParam)
	// Actual dLossDParam is dLossDCombination * dLossDParam
	for i := range dLossDParam {
		dLossDParam[i] *= dLossDCombination
	}

	// Store DCombineDInput into dLossDInput
	n.DCombineDInput(parameters, inputs, combination, dLossDInput)
	// Actual dLossDInput is dLossDCombine * dCombineDInput
	for i := range dLossDInput {
		dLossDInput[i] *= dLossDCombination
	}
}

// Process processes all of the neurons in a layer
// Parameters are the parameters of the neurons,
// inputs are the inputs to that layer (the outputs of the previous layer)
// Stores in place the combinations of the neurons and the outputs of the neurons (as set by neuron.Process)
func ProcessLayer(layer *Layer, parameters [][]float64, inputs []float64, combinations, outputs []float64) {
	for i, neuron := range layer.Neurons {
		combinations[i], outputs[i] = ProcessNeuron(neuron, parameters[i], inputs)
	}
}

// Predict feeds the input through the network and stores the prediction into predOutput.
// It caches the weighted sums and outputs (for example, for use with PredictWithDerivative)
// Assumes the input is appropriately scaled
func Predict(input []float64, net *Net, predOutput []float64, combinations, outputs [][]float64) {
	nLayers := len(net.layers)

	parameters := net.parameters
	layers := net.layers

	// Process the first layer (which uses the input as an input).
	ProcessLayer(&layers[0], parameters[0], input, combinations[0], outputs[0])

	// Process all of the rest of the layers by using the ouptus of the previous layer as inputs
	for i := 1; i < nLayers; i++ {
		ProcessLayer(&layers[i], parameters[i], outputs[i-1], combinations[i], outputs[i])
	}

	// The predicted output is the outputs from the last layer
	// The final set of outputs are the predictions
	copy(predOutput, outputs[nLayers-1])
}

// DerivPredLossTmpMemory is the temporary memory needed for computing the derivative
type PredLossDerivTmpMemory struct {
	combinations [][]float64
	outputs      [][]float64
	dLossDPred   []float64
	dLossDOutput [][]float64
	dLossDInput  [][][]float64
}

// DerivPredLoss predicts the value at the input, compute the value of the loss,
// and computes the derivative of the loss with respect to the parameters
func PredLossDeriv(input []float64, truth []float64, weight float64, net *Net, tmp *PredLossDerivTmpMemory, prediction []float64, dLossDParam [][][]float64) (loss float64) {
	Predict(input, net, prediction, tmp.combinations, tmp.outputs)
	loss = net.Losser.LossAndDeriv(prediction, truth, tmp.dLossDPred)

	// scale the loss and derivative by the weight
	loss *= weight
	floats.Scale(weight, tmp.dLossDPred)
	Derivative(input, net.layers, net.parameters, tmp.dLossDPred, tmp.combinations, tmp.outputs, tmp.dLossDOutput, tmp.dLossDInput, dLossDParam)
	return loss
}

// Find the derivatives of the loss function with respect to the parameters and inputs
// Parameters is all of the parameters of that layer
// Inputs is the input to that layer
// Combinations and outputs are the combinations and outputs for the neurons of that layer
// dLossDOutput is the derivative of the loss with respect to the outputs of that layer
// dLossDParam and dLossDInput are stored in place
func DerivativesLayer(l Layer, parameters [][]float64, inputs []float64, combinations, outputs, dLossDOutput []float64, dLossDParam, dLossDInput [][]float64) {
	for i, neuron := range l.Neurons {
		DLossNeuron(neuron, parameters[i], inputs, combinations[i], outputs[i], dLossDOutput[i], dLossDParam[i], dLossDInput[i])
	}
}

// DInputToDOutput changes the derivative of the loss wrt the inputs of the layer to
// next layer wrt the input and changes it to the derivative of the loss wrt the inputs
// of the previous layer
func DInputToDOutput(nextLayerDLossDInput [][]float64, previousLayerDLossDOutput []float64) {
	for i := range previousLayerDLossDOutput {
		previousLayerDLossDOutput[i] = 0
	}
	// derivative of the loss with respect to the outputs is the sum of its derivatives
	// into the next layers
	for i := range previousLayerDLossDOutput {
		for _, neurDLossDInput := range nextLayerDLossDInput {
			previousLayerDLossDOutput[i] += neurDLossDInput[i]
		}
	}
}

// Derivative computes the derivatives of the loss function with respect to the parameters.
// input, layers, parameters, dLossDPred, combinations, and outputs are all true inputs to the method.
// dLossDParam is the output of the method
// dLossDOutput and dLossDInput are storage for temporary variables
func Derivative(input []float64, layers []Layer, parameters [][][]float64, dLossDPred []float64, combinations, outputs, dLossDOutput [][]float64, dLossDInput, dLossDParam [][][]float64) {
	// For each layer, the following holds
	// dL/dp_{k,i,L} = dL/dout_{i,L} * dout_{i,L}/dcomb_{i,L} * dcomb_{i,L}/dp_{k,i,L}
	// where
	// L is the loss
	// p is the  parameter
	// out is the activation function output
	// comb is the combination function output (i.e. activation input)
	// they are indexed by the kth weight of the ith neuron in the Lth layer.
	// so,
	// dL/dw_{k,i,L} = dLossDWParam,
	// dL/da_{i,L} = dLossDOutput
	// dL/da * da/dS = dLossDCombine
	// dL/da * da/dS * dS/dw = dLossDWeight

	// However, the derivative of the loss with respect to the output of that layer
	// is the sum of the influences of that output on the future outputs
	// This influence depends on the weights
	// Specifically,
	// dL/dout_{i,L-1} = sum_j dL/dcomb_{j, L} * dcomb_{j,L} / dinput_{i,j,L}
	// where input in the ith input
	// note the repetition of the i index in the LHS and the last term of the RHS

	// The derivative of the loss with respect to the ouputs of the last layer
	// is the same as the derivative of the loss function with respect to the
	// predictions (because the outputs of the last layer are the predictions)
	nLayers := len(layers)
	copy(dLossDOutput[nLayers-1], dLossDPred)

	for l := nLayers - 1; l > 0; l-- {
		// Compute dLossDParam and dLossDInput. Inputs to the layer are the outputs of the previous layer
		DerivativesLayer(layers[l], parameters[l], outputs[l-1], combinations[l], outputs[l], dLossDOutput[l], dLossDParam[l], dLossDInput[l])
		for j := range dLossDOutput[l-1] {
			dLossDOutput[l-1][j] = 0
		}
		// Find the derivatives of the outputs for the previous layer
		DInputToDOutput(dLossDInput[l], dLossDOutput[l-1])
	}
	// For the last layer, just need to find the derivative
	DerivativesLayer(layers[0], parameters[0], input, combinations[0], outputs[0], dLossDOutput[0], dLossDParam[0], dLossDInput[0])
}
