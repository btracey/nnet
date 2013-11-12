package nnet

import (
	"errors"
	"fmt"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/scale"
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

	nInputs            int
	nOutputs           int
	totalNumParameters int

	nParameters  [][]int // Number of parameters per neuron
	parameterIdx [][]int // The starting index of the weights of the neuron

	layers          []Layer
	parameters      [][][]float64
	parametersSlice []float64
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
	for i, input := range inputs {
		if len(input) != net.nInputs {
			return nil, fmt.Errorf("Lengths of all the inputs must match net.nInputs. Net inputs: %v, Input %v: %v", net.nInputs, i, len(input))
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

func (net *Net) NewPredLossDerivTmpMemory() *PredLossDerivTmpMemory {
	return &PredLossDerivTmpMemory{
		combinations: net.NewPerNeuronMemory(),
		outputs:      net.NewPerNeuronMemory(),
		dLossDPred:   make([]float64, net.nOutputs),
		dLossDOutput: net.NewPerNeuronMemory(),
		dLossDInput:  net.NewPerInputMemory(),
	}
}
