package nnet

import (
	"github.com/btracey/nnet/activator"
	"math"
	"math/rand"
)

///////////////////////////// Neurons

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

// Set the weights of the neuron to a random value
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
