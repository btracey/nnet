package loss

import (
	"encoding/gob"
	"encoding/json"
	"errors"
	"github.com/btracey/nnet/common"
	"math"
	"reflect"

	//"fmt"
)

func init() {
	gob.Register(SquaredDistance{})
	gob.Register(ManhattanDistance{})
	gob.Register(RelativeSquared(0))
	gob.Register(LogSquared{})

	losserMap = make(map[string]Losser)
	//isPtrMap = make(map[string]bool)

	// Add them to the losser
	Register(&SquaredDistance{})
	Register(&ManhattanDistance{})
	a := RelativeSquared(0)
	Register(&a)
	Register(&LogSquared{})

}

type LossMarshaler struct {
	L Losser
}

type lossMarshaler struct {
	Type  string
	Value Losser
}

type typeUnmarshaler struct {
	Type string
}

type valueUnmarshaler struct {
	Value Losser
}

func (l *LossMarshaler) MarshalJSON() ([]byte, error) {
	name := common.InterfaceFullTypename(l.L)
	// Check that the type has been registered
	_, ok := losserMap[name]
	if !ok {
		return nil, NotRegistered
	}
	marshaler := &lossMarshaler{
		Type:  name,
		Value: l.L,
	}
	return json.Marshal(marshaler)
}
func (l *LossMarshaler) UnmarshalJSON(data []byte) error {
	// First, unmarshal the type
	t := &typeUnmarshaler{}
	json.Unmarshal(data, t)
	// Get the losser
	losser, ok := losserMap[t.Type]
	if !ok {
		return NotRegistered
	}

	v := &valueUnmarshaler{Value: losser}
	json.Unmarshal(data, v)
	l.L = v.Value
	return nil
}

// Losser is an interface for a loss function. It takes in three inputs
// 1) the predicted input value
// 2) the true value
// 3] a receiver for the derivative of the loss function with respect to the prediction
// It returns the value of the loss function itself.
// A loss function in general is a definition of how "bad" a certain prediction is.
//
// GobEncode and GobDecode methods help to save the learning algorithm
type Losser interface {
	LossAndDeriv(prediction []float64, truth []float64, derivative []float64) float64
}

// SquaredDistance is the same as the two-norm of (truth - pred) divided by the length
type SquaredDistance struct{}

// LossAndDeriv computes the average square of the two-norm of (prediction - truth)
// and stores the derivative of the two norm with respect to the prediction in derivative
func (l SquaredDistance) LossAndDeriv(prediction, truth, derivative []float64) (loss float64) {
	for i := range prediction {
		diff := prediction[i] - truth[i]
		derivative[i] = diff
		loss += diff * diff
	}
	loss /= float64(len(prediction))
	for i := range derivative {
		derivative[i] /= float64(len(prediction)) / 2
	}
	return loss
}

// Manhattan distance is the same as the one-norm of (truth - pred)
type ManhattanDistance struct{}

// LossAndDeriv computes the one-norm of (prediction - truth) and stores the derivative of the one norm
// with respect to the prediction in derivative
func (m ManhattanDistance) LossAndDeriv(prediction, truth, derivative []float64) (loss float64) {
	for i := range prediction {
		loss += math.Abs(prediction[i] - truth[i])
		if prediction[i] > truth[i] {
			derivative[i] = 1.0 / float64(len(prediction))
		} else if prediction[i] < truth[i] {
			derivative[i] = -1.0 / float64(len(prediction))
		} else {
			derivative[i] = 0
		}
	}
	loss /= float64(len(prediction))
	return loss
}

// Relative squared is the relative error with the value of RelativeSquared added in the denominator
type RelativeSquared float64

func (r RelativeSquared) LossAndDeriv(prediction, truth, derivative []float64) (loss float64) {
	nSamples := float64(len(prediction))
	for i := range prediction {
		denom := math.Abs(truth[i]) + float64(r)
		diff := prediction[i] - truth[i]

		diffOverDenom := diff / denom

		loss += diffOverDenom * diffOverDenom
		derivative[i] = 2 * diffOverDenom / denom / nSamples
	}
	loss /= nSamples
	return loss
}

// LogSquared uses log(1 + diff*diff) so that really high losses aren't as important
type LogSquared struct{}

func (l LogSquared) LossAndDeriv(prediction, truth, deravitive []float64) (loss float64) {
	nSamples := float64(len(prediction))
	for i := range prediction {
		diff := prediction[i] - truth[i]
		diffSqPlus1 := diff*diff + 1
		loss += math.Log(diffSqPlus1)
		deravitive[i] = 2 / diffSqPlus1 * diff / nSamples
	}
	loss /= nSamples
	return loss
}
