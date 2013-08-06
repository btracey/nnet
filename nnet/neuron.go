package nnet

import (
	"github.com/btracey/nnet/activator"

	"math"
	"math/rand"
)

// netNeuron is the description of the neuron in its context of the full neural net.
// contains a neuron, its ID number and the location of its weights
// in the total weights vector.
//type Neuron struct {
//	Neuroner
//id          int // ID of the neuron (index in the net neuron slice)
//wID         int // ID of the start of the full weight vector
//parameters  []float64
//nParameters int
//}

// Neuron doesn't provide own memory, just a definition. Net interfaces with parameters directly
type Neuron interface {
	NumParameters(nInputs int) int // How many parameters as a function of the number of inputs

	Activate(combination float64) (output float64)
	Combine(parameters, inputs []float64) (combination float64)
	//Parameters() []float64
	//SetParameters([]float64)

	Randomize(parameters []float64) // Set to a random initial condition for doing random restarts

	DActivateDCombination(combination, output float64) (derivative float64)
	DCombineDParameters(params []float64, inputs []float64, combination float64, deriv []float64)
	DCombineDInput(params []float64, inputs []float64, combination float64, deriv []float64)
}

// A sum neuron takes a weighted sum of all the inputs and pipes them through an activator function
type SumNeuron struct {
	activator.Activator
}

// Activate function comes from activator

// NParameters returns the number of parameters
func (s *SumNeuron) NumParameters(nInputs int) int {
	return nInputs + 1
}

// Combine takes a weighted sum of the inputs with the weights set by parameters
// The last element of parameters is the bias term, so len(parameters) = len(inputs) + 1
func (s *SumNeuron) Combine(parameters []float64, inputs []float64) (combination float64) {
	for i, val := range inputs {
		combination += parameters[i] * val
	}
	combination += parameters[len(parameters)-1]
	return
}

// Randomize sets the parameters to a random initial condition
func (s *SumNeuron) Randomize(parameters []float64) {
	for i := range parameters {
		parameters[i] = rand.NormFloat64() * math.Pow(float64(len(parameters)), -0.5)
	}
}

// DActivateDCombination comes from activator

func (s *SumNeuron) DCombineDParameters(params []float64, inputs []float64, combination float64, deriv []float64) {
	// The derivative of the function with respect to the parameters (in this case, the weights), is just
	// the value of the input, and 1 for the bias term
	for i, val := range inputs {
		deriv[i] = val
	}
	deriv[len(deriv)-1] = 1
}

// DCombineDInput Finds the derivative of the combination with respect to the ith input
// The derivative of the combination with respect to the input is the value of the weight
func (s *SumNeuron) DCombineDInput(params []float64, inputs []float64, combination float64, deriv []float64) {
	for i := range inputs {
		deriv[i] = params[i]
	}
	// This intentionally doesn't loop over all of the parameters, as the last parameter is the bias term
}

var (
	TanhNeuron       SumNeuron = SumNeuron{Activator: activator.Tanh{}}
	LinearTanhNeuron SumNeuron = SumNeuron{Activator: activator.LinearTanh{}}
	LinearNeuron     SumNeuron = SumNeuron{Activator: activator.Linear{}}
	SigmoidNeuron    SumNeuron = SumNeuron{Activator: activator.Sigmoid{}}
)

//var TanhNeuron SumNeuron = SumNeuron{Activator: activator.Tanh}

/*
// Neuron is the basic element of the neural net. They take
// in a set of inputs, compute a weighted sum of those inputs
// (using Neuron.Weights), and then computes a function of the
// weighted sum as defined by the activation function
// The final weight is a bias term which is added at the end, so there
// should be one more weight than the number of inputs
type neuron struct {
	weights  []float64
	nWeights int
	activator.Activator
	InputTransform
}

// newNeuron reates a new neuron with the given number of inputs and
// activator function
func newNeuron(nInputs int, a activator.Activator) {
	return &neuron{
		nWeights:  nInputs + 1,
		weights:   make([]float64, nInputs+1),
		Activator: a,
	}
}

// Process computes the weighted sum of inputs and the activation function
func (n *neuron) process(input []float64) (sum, output float64) {
	for i, val := range input {
		sum += val * n.Weights[i]
	}
	sum += n.Weights[n.nWeights-1] //Bias term
	return sum, n.Activate(sum)
}

// randomizeWeights sets the weights of the neuron to a random value
// Uses the randomize proceedure as described in http://leon.bottou.org/slides/tricks/tricks.pdf
func (n *neuron) randomizeWeights() {
	// This specifc equation was chosen to have an expected sum
	// to be between -1 and 1. Assuming the activation function also scales in this
	// range that should be good
	for i := range n.Weights {
		n.Weights[i] = rand.NormFloat64() * math.Pow(float64(n.nWeights), -0.5)

	}
}

// May want GobEncode in the future, still deciding on how we want
// to save and load the net

/*
func (n *Neuron) GobEncode() (buf []byte, err error) {
	w := bytes.NewBuffer(buf)
	encoder := gob.NewEncoder(w)
	err = encoder.Encode(n.Weights)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), err
}

func (n *Neuron) GobDecode(buf []byte) (err error) {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)
	err = decoder.Decode(&n.Weights)
	if err != nil {
		return err
	}
	return err
}
*/
