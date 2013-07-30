package nnet

import (
	"github.com/btracey/nnet/activator"
	"math"
	"math/rand"
)

///////////////////////////// Neurons

// Neurons are the basic element of the neural net. They take
// in a set of inputs, compute a weighted sum of those inputs
// as set by neuron.weights, and then transforms that weighted
// sum into an alternate float64 as defined by the activation function
// The final weight is a bias term which is added at the end, so there
// should be one more weight than the number of inputs
// This type is made public for encoding, but is not intended to be used
// remotely
type Neuron struct {
	Weights  []float64
	nWeights int
	activator.Activator
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

// Compute the weighted sum and the activation function
func (n *Neuron) Process(input []float64) (sum, output float64) {
	for i, val := range input {
		sum += val * n.Weights[i]
	}
	sum += n.Weights[n.nWeights-1] //Bias term
	return sum, n.Activate(sum)
}

// Initialize a neuron with the given number of inputs and the activator
// function
func (n *Neuron) Initialize(nInputs int, r activator.Activator) {
	n.Activator = r
	n.nWeights = nInputs + 1 // Plus one is for the bias term
	n.Weights = make([]float64, n.nWeights)
	// I'm not sure if this should be here or not
	n.RandomizeWeights()
}

// Should the neuron also have a weights randomizer? Probably not.

// Set the weights of the neuron to a random value
func (n *Neuron) RandomizeWeights() {
	// This specifc equation was chosen to have an expected sum
	// to be between -1 and 1. Assuming the activation function also scales in this
	// range that should be good
	for i := range n.Weights {
		n.Weights[i] = rand.NormFloat64() * math.Pow(float64(n.nWeights), -0.5)

	}
}
