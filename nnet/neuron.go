package nnet

import (
	"github.com/btracey/nnet/activator"
	"github.com/btracey/nnet/common"

	"math"
	"math/rand"

	"fmt"

	//"bytes"
	"encoding/gob"
	"encoding/json"
)

func init() {
	gob.Register(&SumNeuron{Activator: activator.LinearTanh{}})
	common.Register(&SumNeuron{})
}

var (
	TanhNeuron       SumNeuron = SumNeuron{Activator: activator.Tanh{}}
	LinearTanhNeuron SumNeuron = SumNeuron{Activator: activator.LinearTanh{}}
	LinearNeuron     SumNeuron = SumNeuron{Activator: activator.Linear{}}
	SigmoidNeuron    SumNeuron = SumNeuron{Activator: activator.Sigmoid{}}
)

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
	//gob.GobDecoder
	//gob.GobEncoder

	// Shouldn't need to add json.Marshaler because layer can do it
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

type activatorMarshaler struct {
	Activator *common.InterfaceMarshaler
}

func (s *SumNeuron) MarshalJSON() (b []byte, err error) {
	n := &activatorMarshaler{Activator: &common.InterfaceMarshaler{I: s.Activator}}
	return json.Marshal(n)
}

func (s *SumNeuron) UnmarshalJSON(data []byte) error {
	v := &activatorMarshaler{}
	err := json.Unmarshal(data, v)
	if err != nil {
		return fmt.Errorf("nnet/neuron/unmarshaljson: error unmarshaling data: " + err.Error())
	}
	s.Activator = v.Activator.I.(activator.Activator)
	return nil
}
