package activator

import "math"

// A set of built-in Activator types

// Activator is an interface for the activation function of the neuron,
// (allowing for neurons with custom activation functions).
// An activator has two methods.
// 1) Activate, which is the actual activation function, taking in the
// weighted sum of the inputs and outputing the resulting value
// 2) DActivateDSum which is the derivative of the activation function
// with respect to the sum. DActivateDSum takes in two arguments,
// the weighted sum itself and the output of Activate. This complication
// arises from the fact that some derivatives are significantly easier
// to compute given one value or the other.
type Activator interface {
	Activate(sum float64) float64
	DActivateDSum(sum float64, output float64) float64
}

// A sigmoid neuron has a sigmoid as the activation function
// out = 1/(1 + exp(-sum))
type Sigmoid struct{}

// Computes the sigmoid activation function
func (a Sigmoid) Activate(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

// Computes the derivative of the activation function
func (n Sigmoid) DActivateDSum(sum, output float64) float64 {
	return output * (1 - output)
}

// Linear neuron has a linear activation function out = sum
type Linear struct{}

// Computes the linear activation function
func (a Linear) Activate(sum float64) float64 {
	return sum
}

// Computes the derivative of the linear activation function
func (a Linear) DActivateDSum(sum, output float64) float64 {
	return 1.0
}

// Tanh has a tanh activation function. The constants are set to have a
// range between -1 and 1
type Tanh struct{}

// Computes the Tanh activation function
func (a Tanh) Activate(sum float64) float64 {
	return 1.7159 * math.Tanh(2.0/3.0*sum)
}

const (
	// http://www.wolframalpha.com/input/?i=1.7159+*+2%2F3
	TanhConst = 1.14393333333333333333333333333333333333333333333333333333333333333333
	TwoThirds = 0.66666666666666666666666666666666666666666666666666666666666666666666
)

//Computes the derivative of the Tanh activation function
func (a Tanh) DActivateDSum(sum, output float64) float64 {
	return TanhConst * (1.0 - math.Tanh(TwoThirds*sum)*math.Tanh(TwoThirds*sum))
}
