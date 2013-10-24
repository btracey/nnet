package activator

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"testing"
)

type MarshalActivator interface {
	Activator
	json.Marshaler
}

type UnmarshalActivator interface {
	Activator
	json.Unmarshaler
}

func JSONTest(first MarshalActivator, second UnmarshalActivator) error {
	b, err := first.MarshalJSON()
	if err != nil {
		return fmt.Errorf("Error marshaling")
	}
	err = second.UnmarshalJSON(b)
	if err != nil {
		return fmt.Errorf("Error unmarshaling")
	}
	if !(reflect.DeepEqual(first, reflect.ValueOf(second).Elem().Interface())) {
		return fmt.Errorf("Unequal after unmarshal")
	}

	// Check package level function
	b2, err := MarshalJSON(first)
	if err != nil {
		return fmt.Errorf("Error using package marshal: " + err.Error())
	}

	newActivator, err := UnmarshalJSON(b2)
	if err != nil {
		return fmt.Errorf("Error using package unmarshal: " + err.Error())
	}

	if !reflect.DeepEqual(newActivator, first) {
		return fmt.Errorf("Not equal after package unmarshal")
	}
	return nil
}

// TODO: Add better tests for JSON

func TestSigmoid(t *testing.T) {
	s := Sigmoid{}
	sum := 1.23456789
	// const gotten from wolfram alpha
	// http://www.wolframalpha.com/input/?i=1%2F%281%2B+exp%28-1.23456789%29%29
	// http://www.wolframalpha.com/input/?i=e%5E1.23456789%2F%281%2Be%5E1.23456789%29%5E2
	const trueOut = 0.7746170617399426534330751940207841531562387807774740
	const trueDeriv = 0.17458546940132052486381952968243494067770801388762
	output := s.Activate(sum)
	if math.Abs(output-float64(trueOut)) > 1E-15 {
		t.Errorf("Activation output does not match. %v expected, %v found", trueOut, output)
	}
	deriv := s.DActivateDCombination(sum, output)
	if math.Abs(deriv-float64(trueDeriv)) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
	}

	err := JSONTest(Sigmoid{}, &Sigmoid{})
	if err != nil {
		t.Errorf("Error using JSON: " + err.Error())
	}
}

func TestLinear(t *testing.T) {
	s := Linear{}
	sum := 1.23456789
	trueOut := sum
	trueDeriv := 1.0
	output := s.Activate(sum)
	if math.Abs(output-trueOut) > 1E-15 {
		t.Errorf("Activation output does not match. %v expected, %v found", trueOut, output)
	}
	deriv := s.DActivateDCombination(sum, output)
	if math.Abs(deriv-trueDeriv) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
	}

	err := JSONTest(Linear{}, &Linear{})
	if err != nil {
		t.Errorf("Error using JSON: " + err.Error())
	}

}

func TestTanh(t *testing.T) {
	s := Tanh{}
	sum := 1.23456789
	// const gotten from wolfram alpha
	// http://www.wolframalpha.com/input/?i=1.7159+*+tanh%282%2F3+*+1.23456789%29
	// http://www.wolframalpha.com/input/?i=1.14393333333333333333333333333333333333333333333333333333333333333333+*+sech%5E2%28%282+*+1.23456789%29%2F3%29
	const trueOut = 1.1611906180541956173946145965239159139732343741372935
	const trueDeriv = 0.620063002328385566791528134365391691015175805527057181714408
	output := s.Activate(sum)
	if math.Abs(output-float64(trueOut)) > 1E-15 {
		t.Errorf("Activation output does not match. %v expected, %v found", trueOut, output)
	}
	deriv := s.DActivateDCombination(sum, output)
	if math.Abs(deriv-float64(trueDeriv)) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
	}
	err := JSONTest(Tanh{}, &Tanh{})
	if err != nil {
		t.Errorf("Error using JSON: " + err.Error())
	}
}

func TestLinearTanh(t *testing.T) {
	s := LinearTanh{}
	sum := 1.23456789
	// const gotten from wolfram alpha
	// http://www.wolframalpha.com/input/?i=1.7159+*+tanh%282%2F3+*+1.23456789%29+%2B+0.01+*+1.23456789
	// http://www.wolframalpha.com/input/?i=1.14393333333333333333333333333333333333333333333333333333333333333333+*+sech%5E2%28%282+*+1.23456789%29%2F3%29+%2B+0.01
	const trueOut = 1.1735362969541956173946145965239159139732343741372935
	const trueDeriv = 0.6300630023283855667915281343653916910151758055270571
	output := s.Activate(sum)
	if math.Abs(output-float64(trueOut)) > 1E-15 {
		t.Errorf("Activation output does not match. %v expected, %v found", trueOut, output)
	}
	deriv := s.DActivateDCombination(sum, output)
	if math.Abs(deriv-float64(trueDeriv)) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
	}

	err := JSONTest(LinearTanh{}, &LinearTanh{})
	if err != nil {
		t.Errorf("Error using JSON: " + err.Error())
	}
}
