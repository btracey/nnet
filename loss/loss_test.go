package loss

import (
	//"fmt"
	//"encoding/json"
	"errors"
	"github.com/btracey/nnet/common"
	"github.com/gonum/floats"
	"math"
	"reflect"
	"testing"
)

const (
	FDStep = 10E-6
	FDTol  = 10E-8
	TOL    = 1E-14
)

var doesNotMatch error = errors.New("Does not match")

func testMarshalAndUnmarshal(l Losser) error {
	l1 := &common.InterfaceMarshaler{I: l}
	b, err := l1.MarshalJSON()
	if err != nil {
		return err
	}
	l2 := &common.InterfaceMarshaler{}
	err = l2.UnmarshalJSON(b)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(l, l2.I) {
		return doesNotMatch
	}
	return nil
}

func finiteDifferenceLosser(losser Losser, prediction, truth []float64) (derivative, fdDerivative []float64) {
	if len(prediction) != len(truth) {
		panic("prediction and truth are not the same length")
	}
	derivative = make([]float64, len(prediction))
	losser.LossAndDeriv(prediction, truth, derivative)

	fdDerivative = make([]float64, len(prediction))
	newDerivative1 := make([]float64, len(prediction))
	newDerivative2 := make([]float64, len(prediction))
	for i := range prediction {
		prediction[i] += FDStep
		loss1 := losser.LossAndDeriv(prediction, truth, newDerivative1)
		prediction[i] -= 2 * FDStep
		loss2 := losser.LossAndDeriv(prediction, truth, newDerivative2)
		prediction[i] += FDStep
		fdDerivative[i] = (loss1 - loss2) / (2 * FDStep)
	}
	return
}

func TestSquaredDistance(t *testing.T) {
	prediction := []float64{1, 2, 3}
	truth := []float64{1.1, 2.2, 2.7}
	trueloss := (.1*.1 + .2*.2 + .3*.3) / 3
	derivative := []float64{0, 0, 0}

	sq := SquaredDistance{}
	loss := sq.LossAndDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("Loss doesn't match")
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. deriv: %v, fdDeriv: %v ", derivative, fdDerivative)
	}

	err := testMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}

func TestManhattanDistance(t *testing.T) {
	prediction := []float64{1, 2, 3}
	truth := []float64{1.1, 2.2, 2.7}
	trueloss := (.1 + .2 + .3) / 3
	derivative := []float64{0, 0, 0}

	sq := ManhattanDistance{}
	loss := sq.LossAndDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("Loss doesn't match. %v found, %v expected", loss, trueloss)
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}

	err := testMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}

func TestRelativeSquared(t *testing.T) {
	tol := 1e-2
	prediction := []float64{1, -2, 3}
	truth := []float64{1.1, -2.2, 2.7}
	trueloss := ((.1/(1.1+tol))*(.1/(1.1+tol)) + (.2/(2.2+tol))*(.2/(2.2+tol)) + (.3/(2.7+tol))*(.3/(2.7+tol))) / 3
	derivative := []float64{0, 0, 0}

	sq := RelativeSquared(tol)
	loss := sq.LossAndDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("Loss doesn't match. %v found, %v expected", loss, trueloss)
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}

	err := testMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}

func TestLogSquared(t *testing.T) {
	prediction := []float64{1, -2, 3}
	truth := []float64{1.1, -2.2, 2.7}
	trueloss := (math.Log(.1*.1+1) + math.Log(.2*.2+1) + math.Log(.3*.3+1)) / 3
	derivative := []float64{0, 0, 0}

	sq := LogSquared{}
	loss := sq.LossAndDeriv(prediction, truth, derivative)
	if math.Abs(loss-trueloss) > TOL {
		t.Errorf("Loss doesn't match. %v found, %v expected", loss, trueloss)
	}
	derivative, fdDerivative := finiteDifferenceLosser(sq, prediction, truth)
	if !floats.EqualApprox(derivative, fdDerivative, FDTol) {
		t.Errorf("Derivative doesn't match. \n deriv: %v \n fdDeriv: %v ", derivative, fdDerivative)
	}
	err := testMarshalAndUnmarshal(sq)
	if err != nil {
		t.Errorf("Error marshaling and unmarshaling")
	}
}
