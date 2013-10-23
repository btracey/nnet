package loss

import (
	"encoding/gob"
	"encoding/json"
	"errors"
	"github.com/btracey/nnet/common"
	"math"
)

func init() {
	gob.Register(SquaredDistance{})
	gob.Register(ManhattanDistance{})
	gob.Register(RelativeSquared(0))
	gob.Register(LogSquared{})
}

// MarshalText marshalls the activators in this package for use with a
// TextMarshaller. If the activator is not from this package, a
// NotInPackage error will be returned
func MarshalJSON(l Losser) ([]byte, error) {
	// New types added with this package should be added here.
	// Types should use prefix while marshalling
	switch l.(type) {
	default:
		return nil, NotInPackage
	case SquaredDistance, ManhattanDistance, RelativeSquared, LogSquared:
		t := l.(json.Marshaler)
		return t.MarshalJSON()
	}
}

// UnmarshalText returns a package activator from the string.
// If the string does not match any of the types in the package,
// then a "NotInPackage" error is returned
func UnmarshalJSON(b []byte) (Losser, error) {
	tmp := string(b)
	// Remove quotations
	str := tmp[len("\"") : len(tmp)-len("\"")]

	// TODO: Add in something about relative squared

	switch str {
	case sqDistString:
		return SquaredDistance{}, nil
	case manhatDistString:
		return ManhattanDistance{}, nil
	//case relSqString:
	//	return RelativeSquared{}, nil
	case logSqString:
		return LogSquared{}, nil
	default:
		return nil, common.NotInPackage
	}
}

var NotInPackage = errors.New("NotInPackage")
var UnmarshallMismatch = errors.New("Unmarshal string mismatch")

// prefix is for marshalling and unmarshalling. The
var prefix string = "github.com/btracey/nnet/loss"

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

var sqDistString string = "SquaredDistance"

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

// MarshalJSON marshals the sigmoid into UTF-8 text
func (a SquaredDistance) MarshalJSON() ([]byte, error) {
	return json.Marshal(sigmoidMarshalString)
}

// MarshalJSON marshals the sigmoid into UTF-8 text
func (a *SquaredDistance) UnmarshalJSON(input []byte) error {
	var str string
	json.Unmarshal(input, &str)
	if str != sigmoidMarshalString {
		return common.UnmarshalMismatch{Expected: sigmoidMarshalString, Received: str}
	}
	a = &SquaredDistance{}
	return nil
}

var manhatDistString string = "ManhattanDistance"

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

var relSqString string = "RelativeSquared"

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

var logSqString string = "LogSquared"

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

/*
// Information treats the predictions as coming from the
// normal distribution. Assumes that the distribution
// is unimodal
type NormalInformation struct{}

func (n NormalInformation) LossAndDeriv(prediction, truth, derivative []float64) {
	nDim := float64(len(prediction))
	for i := range prediction{
		// If both prediction and the truth are on the same side of the mode,
		// just find the difference in their log probability
		if (prediction < 0 && truth < 0 ) || (truth > 0 && prediction > 0){

		}
	}
}
*/
