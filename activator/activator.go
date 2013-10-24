package activator

import (
	"encoding/gob"
	"encoding/json"
	"github.com/btracey/nnet/common"
	"math"
)

// init registers the types so they can be GobEncoded and GobDecoded
func init() {
	gob.Register(Sigmoid{})
	gob.Register(Linear{})
	gob.Register(Tanh{})
	gob.Register(LinearTanh{})
}

// prefix is for marshalling and unmarshalling. The
var prefix string = "github.com/btracey/nnet/activator/"

//
type marshalName struct{ Name string }

// MarshalText marshalls the activators in this package for use with a
// TextMarshaller. If the activator is not from this package, a
// NotInPackage error will be returned
func MarshalJSON(a Activator) ([]byte, error) {
	// New types added with this package should be added here.
	// Types should use prefix while marshalling
	switch a.(type) {
	default:
		return nil, common.NotInPackage
	case Sigmoid, Linear, Tanh, LinearTanh:
		t := a.(json.Marshaler)
		return t.MarshalJSON()
	}
}

// UnmarshalText returns a package activator from the string.
// If the string does not match any of the types in the package,
// then a "NotInPackage" error is returned
func UnmarshalJSON(b []byte) (Activator, error) {

	name := &marshalName{}
	json.Unmarshal(b, name)
	switch name.Name {
	case sigmoidMarshalString:
		return Sigmoid{}, nil
	case linearMarshalString:
		return Linear{}, nil
	case tanhMarshalString:
		return Tanh{}, nil
	case linearTanhMarshalString:
		return LinearTanh{}, nil
	default:
		return nil, common.NotInPackage
	}
}

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
	DActivateDCombination(sum float64, output float64) float64
}

// Sigmoid is an activation function which is the sigmoid function,
// out = 1/(1 + exp(-sum))
type Sigmoid struct{}

// Here so that if changed, it is in one place here and in the Unmarshal code
var sigmoidString string = "Sigmoid"

// Activate computes the sigmoid activation function
func (a Sigmoid) Activate(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

// DActivateDSum computes the derivative of the activation function
// with respect to the weighted sum
func (n Sigmoid) DActivateDCombination(sum, output float64) float64 {
	return output * (1 - output)
}

/*
func (a Sigmoid) String() string {
	return sigmoidString
}
*/

var sigmoidMarshalString string = prefix + sigmoidString

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a Sigmoid) MarshalJSON() ([]byte, error) {
	return json.Marshal(marshalName{Name: sigmoidMarshalString})
}

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a *Sigmoid) UnmarshalJSON(input []byte) error {
	s := &marshalName{}
	json.Unmarshal(input, &s)
	if s.Name != sigmoidMarshalString {
		return common.UnmarshalMismatch{Expected: sigmoidMarshalString, Received: s.Name}
	}
	a = &Sigmoid{}
	return nil
}

// Linear neuron has a the identity activation function out = sum
type Linear struct{}

// Here so that if changed, it is in one place here and in the Unmarshal code
var linearString string = "Linear"

// Activate computes the linear activation function
func (a Linear) Activate(sum float64) float64 {
	return sum
}

// DActivateDSum computes the derivative of the linear activation function
// with respect to the weighted sum
func (a Linear) DActivateDCombination(sum, output float64) float64 {
	return 1.0
}

func (a Linear) String() string {
	return linearString
}

var linearMarshalString string = prefix + linearString

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a Linear) MarshalJSON() ([]byte, error) {
	return json.Marshal(marshalName{Name: linearMarshalString})
}

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a *Linear) UnmarshalJSON(input []byte) error {
	s := &marshalName{}
	json.Unmarshal(input, &s)
	if s.Name != linearMarshalString {
		return common.UnmarshalMismatch{Expected: linearMarshalString, Received: s.Name}
	}
	a = &Linear{}
	return nil
}

const (
	// http://www.wolframalpha.com/input/?i=1.7159+*+2%2F3
	TanhDerivConst = 1.14393333333333333333333333333333333333333333333333333333333333333333
	TwoThirds      = 0.66666666666666666666666666666666666666666666666666666666666666666666
)

// Source for tanh activation function:

// Tanh has a tanh activation function. out = a tanh(b * sum). The constants
// a and b are set so that tanh has a value of -1 and 1 when the sum = -1 and 1
// respectively.
// See: http://leon.bottou.org/slides/tricks/tricks.pdf for more description
type Tanh struct{}

// Here so that if changed, it is in one place here and in the Unmarshal code
var tanhString string = "Tanh"

// Activate computes the Tanh activation function
func (a Tanh) Activate(sum float64) float64 {
	return 1.7159 * math.Tanh(2.0/3.0*sum)
}

// DActivateDSum computes the derivative of the Tanh activation function
// with respect to the weighted sum
func (a Tanh) DActivateDCombination(sum, output float64) float64 {
	return TanhDerivConst * (1.0 - math.Tanh(TwoThirds*sum)*math.Tanh(TwoThirds*sum))
}

func (a Tanh) String() string {
	return tanhString
}

var tanhMarshalString string = prefix + tanhString

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a Tanh) MarshalJSON() ([]byte, error) {
	return json.Marshal(marshalName{Name: tanhMarshalString})
}

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a *Tanh) UnmarshalJSON(input []byte) error {
	s := &marshalName{}
	json.Unmarshal(input, &s)
	if s.Name != tanhMarshalString {
		return common.UnmarshalMismatch{Expected: tanhMarshalString, Received: s.Name}
	}
	a = &Tanh{}
	return nil
}

// Source for linear tanh activation function: http://leon.bottou.org/slides/tricks/tricks.pdf

// LinearTahn is the Tanh activation function plus a small linear term (set to 0.01).
// This linear term helps stabilize the weights so that they do not tend to infinity.
// See: // See: http://leon.bottou.org/slides/tricks/tricks.pdf for more description
type LinearTanh struct {
}

// Here so that if changed, it is in one place here and in the Unmarshal code
var linearTanhString string = "LinearTanh"

// Activate computes the LinearTanh activation function
func (a LinearTanh) Activate(sum float64) float64 {
	return 1.7159*math.Tanh(2.0/3.0*sum) + 0.01*sum
}

// DActivateDSum computes the derivative of the Tanh activation function
// with respect to the weighted sum
func (a LinearTanh) DActivateDCombination(sum, output float64) float64 {
	return TanhDerivConst*(1.0-math.Tanh(TwoThirds*sum)*math.Tanh(TwoThirds*sum)) + 0.01
}

func (a LinearTanh) String() string {
	return linearTanhString
}

var linearTanhMarshalString string = prefix + linearTanhString

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a LinearTanh) MarshalJSON() ([]byte, error) {
	return json.Marshal(marshalName{Name: linearTanhMarshalString})
}

// MarshalJSON marshalls the sigmoid into UTF-8 text
func (a *LinearTanh) UnmarshalJSON(input []byte) error {
	s := &marshalName{}
	json.Unmarshal(input, &s)
	if s.Name != linearTanhMarshalString {
		return common.UnmarshalMismatch{Expected: linearTanhMarshalString, Received: s.Name}
	}
	a = &LinearTanh{}
	return nil
}
