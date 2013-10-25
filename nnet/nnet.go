package nnet

import (
	"github.com/btracey/nnet/common"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/scale"

	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	//"reflect"
	"sync"
)

type SaveFormat int

const (
	Gob SaveFormat = iota
	Ascii
)

// Net is the structure representing a feed-forward artificial network.
// Parameters and ParametersVector are accessible for training purposes,
// but they should not be resliced or appended to. The memories are linked
type Net struct {
	Losser       loss.Losser  // The loss function for training (needed for computing the derivative)
	InputScaler  scale.Scaler // The way in which the data should be scaled (and unscaled)
	OutputScaler scale.Scaler // The way in which the data should be scaled (and unscaled)
	//NeuronActivator activator.Activator
	//FinalActivator  activator.Activator

	nInputs            int
	nOutputs           int
	totalNumParameters int

	nParameters  [][]int // Number of parameters per neuron
	parameterIdx [][]int // The starting index of the weights of the neuron

	layers          []Layer
	parameters      [][][]float64
	parametersSlice []float64
}

func (net *Net) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	var err error
	encoder := gob.NewEncoder(w)
	err = encoder.Encode(&net.Losser)
	if err != nil {
		return nil, fmt.Errorf("Error encoding Losser: %v", err)
	}

	err = encoder.Encode(&net.InputScaler)
	if err != nil {
		return nil, fmt.Errorf("Error encoding input scaler: %v", err)
	}

	err = encoder.Encode(&net.OutputScaler)
	if err != nil {
		return nil, fmt.Errorf("Error encoding output scaler: %v", err)
	}

	err = encoder.Encode(net.nInputs)
	if err != nil {
		return nil, fmt.Errorf("Error encoding nInputs: %v", err)
	}

	err = encoder.Encode(net.layers)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(&net.parameters)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

type netMarshal struct {
	Losser                 *common.InterfaceMarshaler
	InputScaler            *common.InterfaceMarshaler
	OutputScaler           *common.InterfaceMarshaler
	NumInputs              int
	NumOutputs             int
	TotalNumParameters     int
	NumParametersPerNeuron [][]int
	ParameterIndex         [][]int
}

// Save the net to a string file (for reading in to non-go programs for example). If custom
// interfaces are used, they will be marshaled with custom and if the custom type is a text
// marshaller it will write
func (net *Net) MarshalJSON() (b []byte, err error) {
	// First, martial the interfaces
	n := &netMarshal{

		Losser: &common.InterfaceMarshaler{Value: net.Losser},
		/*
			InputScaler:            &common.InterfaceMarshaler{Value: net.InputScaler},
			OutputScaler:           &common.InterfaceMarshaler{Value: net.OutputScaler},
		*/
		NumInputs:              net.nInputs,
		NumOutputs:             net.nOutputs,
		TotalNumParameters:     net.totalNumParameters,
		NumParametersPerNeuron: net.nParameters,
		ParameterIndex:         net.parameterIdx,
	}
	return json.Marshal(n)
}

// TextUnmarshaler
func (net *Net) UnmarshalJSON(text []byte) error {
	return nil
}

// GobDecode some comment about needing to register custom types
func (net *Net) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)

	var err error

	err = decoder.Decode(&net.Losser)
	if err != nil {
		return fmt.Errorf("Error decoding losser: %v", err)
	}

	err = decoder.Decode(&net.InputScaler)
	if err != nil {
		return fmt.Errorf("Error decoding input scaler: %v", err)
	}

	err = decoder.Decode(&net.OutputScaler)
	if err != nil {
		return fmt.Errorf("Error decoding output scaler: %v", err)
	}
	err = decoder.Decode(&net.nInputs)
	if err != nil {
		return fmt.Errorf("Error decoding nInputs: %v", err)
	}

	err = decoder.Decode(&net.layers)
	if err != nil {
		return fmt.Errorf("Error decoding layers: %v", err)
	}
	net.new()
	err = decoder.Decode(&net.parameters)
	if err != nil {
		return fmt.Errorf("Error decoding parameters: %v", err)
	}

	return nil
}

// Save saves the neural net
func (net *Net) Save(filename string) error {
	bytes, err := net.GobEncode()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, bytes, 0700)
}

// Load loads in a neural net from a file.
func Load(filename string) (*Net, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	net := &Net{}
	err = net.GobDecode(bytes)
	if err != nil {
		return nil, err
	}
	return net, nil
}

// new fills a net that already has the nInputs and the layers specified
func (net *Net) new() {
	layers := net.layers
	nInputs := net.nInputs
	totalNumParameters := 0
	nParameters := make([][]int, len(layers))
	parameterIdx := make([][]int, len(layers))
	// Count up the number of parameters
	for i, layer := range layers {
		nParameters[i] = make([]int, len(layer.Neurons))
		parameterIdx[i] = make([]int, len(layer.Neurons))
		for j, neuron := range layer.Neurons {
			var neuronParams int
			if i == 0 {
				// If it is the first layer...
				neuronParams = neuron.NumParameters(nInputs)
			} else {
				neuronParams = neuron.NumParameters(len(layers[i-1].Neurons))
			}
			nParameters[i][j] = neuronParams
			parameterIdx[i][j] = totalNumParameters
			totalNumParameters += neuronParams
		}
	}
	net.totalNumParameters = totalNumParameters
	net.parameterIdx = parameterIdx
	net.nParameters = nParameters
	// Make memory for all the parameters (we want a vector to allow easy training)
	net.parametersSlice = make([]float64, totalNumParameters)

	// Reslice it to make it a slice of slice of slices
	net.parameters = make([][][]float64, len(layers))
	for i, layer := range layers {
		net.parameters[i] = make([][]float64, len(layer.Neurons))
		for j := range layer.Neurons {
			net.parameters[i][j] = net.parametersSlice[net.parameterIdx[i][j] : net.parameterIdx[i][j]+net.nParameters[i][j]]
		}
	}
	net.nOutputs = len(layers[len(layers)-1].Neurons)
	net.RandomizeParameters()
}

// NewNet creates a new net
func NewNet(nInputs int, layers []Layer) *Net {
	net := &Net{
		layers:  layers,
		nInputs: nInputs,
	}
	net.new()
	return net
}

// Inputs returns the number of inputs in the net
func (net *Net) Inputs() int {
	return net.nInputs
}

// Outputs returns the number of outputs in the net
func (net *Net) Outputs() int {
	return net.nOutputs
}

// TotalNumParameters returns the total number of parameters in the net
func (net *Net) TotalNumParameters() int {
	return net.totalNumParameters
}

// RandomizeParameters randomizes the parameters of the net
func (net *Net) RandomizeParameters() {
	for i, layer := range net.layers {
		for j, neuron := range layer.Neurons {
			neuron.Randomize(net.parameters[i][j])
		}
	}
}

// ParametersSlice copies the parameters into dst
func (net *Net) ParametersSlice(dst []float64) {
	if len(dst) != net.totalNumParameters {
		panic("length of dst does not match len of net")
	}
	copy(dst, net.parametersSlice)
}

// SetParametersSlice sets the paramaters to the values in src
func (net *Net) SetParametersSlice(src []float64) {
	if len(src) != net.totalNumParameters {
		panic("length of src does not match len of net")
	}
	copy(net.parametersSlice, src)
}

// MakeNeuronMemory creates new memory with one value per
// neuron in the net (indexed by layer and then neuron)
func (net *Net) NewPerNeuronMemory() [][]float64 {
	mem := make([][]float64, len(net.layers))
	for i, layer := range net.layers {
		mem[i] = make([]float64, len(layer.Neurons))
	}
	return mem
}

// MakeParameterMemory creates new memory with one value
// per parameter in the net. Indexed by layer then neuron
// then parameter
func (net *Net) NewPerParameterMemory() (tiered [][][]float64, flat []float64) {
	count := 0
	flat = make([]float64, net.totalNumParameters)
	tiered = make([][][]float64, len(net.layers))
	for i, layer := range net.layers {
		tiered[i] = make([][]float64, len(layer.Neurons))
		for j := range layer.Neurons {
			tiered[i][j] = flat[count : count+net.nParameters[i][j]]
			count += net.nParameters[i][j]
		}
	}
	return
}

// MakeInputMemory creates new memory with one value per
// input to each neuron
func (net *Net) NewPerInputMemory() [][][]float64 {
	mem := make([][][]float64, len(net.layers))
	for i, layer := range net.layers {
		mem[i] = make([][]float64, len(layer.Neurons))
	}
	// First layer has number of inputs equal to the number
	// of inputs of the net
	for i := range net.layers[0].Neurons {
		mem[0][i] = make([]float64, net.nInputs)
	}
	// All other layers have a number of inputs equal to the
	// number of nodes in the previous layer
	for i := 1; i < len(net.layers); i++ {
		for j := range net.layers[i].Neurons {
			mem[i][j] = make([]float64, len(net.layers[i-1].Neurons))
		}
	}
	return mem
}

type InputMismatch struct {
	Provided int
	Expected int
}

func (i InputMismatch) Error() string {
	return fmt.Sprintf("Length of input must match the number of inputs of the net. %d inputs prodived, but the net has %d inputs", i.Provided, i.Expected)
}

// Predict predicts the value at the input location. Panics if
// len(input) != net.NumInputs() and if len(output) != net.NumOutputs()
func (net *Net) Predict(input []float64) (pred []float64, err error) {
	if len(input) != net.nInputs {
		return nil, InputMismatch{Provided: len(input), Expected: net.nInputs}
	}

	predOutput := make([]float64, net.nOutputs)
	predictTmpMemory := net.NewPredictTmpMemory()

	if !net.InputScaler.IsScaled() {
		return nil, errors.New("Scale must be set before calling predict")
	}
	if !net.OutputScaler.IsScaled() {
		return nil, errors.New("Scale must be set before calling predict")
	}

	net.InputScaler.Scale(input)
	defer net.InputScaler.Unscale(input)

	Predict(input, net, predOutput, predictTmpMemory.combinations, predictTmpMemory.outputs)
	defer net.OutputScaler.Unscale(predOutput)
	return predOutput, nil
}

func (net *Net) PredictSlice(inputs [][]float64) (predictions [][]float64, err error) {

	if !net.InputScaler.IsScaled() {
		return nil, errors.New("Scale must be set before calling predict")
	}
	if !net.OutputScaler.IsScaled() {
		return nil, errors.New("Scale must be set before calling predict")
	}
	for _, input := range inputs {
		if len(input) != net.nInputs {
			return nil, errors.New("Lengths of all the inputs must match net.nInputs")
		}
	}
	predictions = make([][]float64, len(inputs))
	for i := range predictions {
		predictions[i] = make([]float64, net.nOutputs)
	}

	err = scale.ScaleData(net.InputScaler, inputs)
	if err != nil {
		return nil, errors.New("Error using ScaleData: " + err.Error())
	}
	defer scale.UnscaleData(net.InputScaler, inputs)

	w := sync.WaitGroup{}
	// Predict samples in parallel
	chunkSize := 100
	count := 0
	var endInd int
	for {
		if count+chunkSize > len(inputs) {
			endInd = len(inputs)
		} else {
			endInd = count + chunkSize
		}
		w.Add(1)
		go func(startInd, endInd int) {
			p := net.NewPredictTmpMemory()
			for i := startInd; i < endInd; i++ {
				Predict(inputs[i], net, predictions[i], p.combinations, p.outputs)
			}
			w.Done()
		}(count, endInd)
		if endInd == len(inputs) {
			break
		}
		count += chunkSize
	}
	w.Wait()
	defer scale.UnscaleData(net.OutputScaler, predictions)
	return predictions, nil
}

type PredictTmpMemory struct {
	combinations [][]float64
	outputs      [][]float64
}

func (net *Net) NewPredictTmpMemory() *PredictTmpMemory {
	return &PredictTmpMemory{
		combinations: net.NewPerNeuronMemory(),
		outputs:      net.NewPerNeuronMemory(),
	}
}

// DefaultRegression returns the default network for regression problems of the given size
// nHiddenLayers must be at least 1, and nNeuronsPerLayer must be at least zero
func DefaultRegression(nInputs, nOutputs, nHiddenLayers, nNeuronsPerHiddenLayer int) *Net {
	if nHiddenLayers < 1 {
		panic("number of hidden layers must be at least 1")
	}
	if nNeuronsPerHiddenLayer < 1 {
		panic("number of neurons per layer must be at least 1")
	}
	if nInputs < 1 {
		panic("number of inputs must be at least 1")
	}
	if nOutputs < 1 {
		panic("number of outputs must be at least 1")
	}
	// The +1 is for the output layer
	layers := make([]Layer, nHiddenLayers+1)
	// Make memory for neurons and have all of the hidden layer neurons have a linear tanh
	// activation function
	for i := 0; i < nHiddenLayers; i++ {
		layers[i].Neurons = make([]Neuron, nNeuronsPerHiddenLayer)
		for j := range layers[i].Neurons {
			layers[i].Neurons[j] = &TanhNeuron
		}
	}
	// All of the output layers should be linear (for regression)
	layers[len(layers)-1].Neurons = make([]Neuron, nOutputs)
	for j := range layers[len(layers)-1].Neurons {
		layers[len(layers)-1].Neurons[j] = &LinearNeuron
	}
	net := NewNet(nInputs, layers)
	net.Losser = loss.SquaredDistance{}
	net.InputScaler = &scale.Normal{}
	net.OutputScaler = &scale.Normal{}
	return net
}

// layer represents a layer of neurons
type Layer struct {
	Neurons []Neuron
}

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

func (net *Net) NewPredLossDerivTmpMemory() *PredLossDerivTmpMemory {
	return &PredLossDerivTmpMemory{
		combinations: net.NewPerNeuronMemory(),
		outputs:      net.NewPerNeuronMemory(),
		dLossDPred:   make([]float64, net.nOutputs),
		dLossDOutput: net.NewPerNeuronMemory(),
		dLossDInput:  net.NewPerInputMemory(),
	}
}

// DerivPredLoss predicts the value at the input, compute the value of the loss,
// and computes the derivative of the loss with respect to the parameters
func PredLossDeriv(input []float64, truth []float64, net *Net, tmp *PredLossDerivTmpMemory, prediction []float64, dLossDParam [][][]float64) (loss float64) {
	Predict(input, net, prediction, tmp.combinations, tmp.outputs)
	loss = net.Losser.LossAndDeriv(prediction, truth, tmp.dLossDPred)
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
