package activator

import (
	"math"
	"testing"
)

// Test to make sure the activator functions work as they should

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
	deriv := s.DActivateDSum(sum, output)
	if math.Abs(deriv-float64(trueDeriv)) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
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
	deriv := s.DActivateDSum(sum, output)
	if math.Abs(deriv-trueDeriv) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
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
	deriv := s.DActivateDSum(sum, output)
	if math.Abs(deriv-float64(trueDeriv)) > 1E-15 {
		t.Errorf("Derivative does not match. %v expected, %v found", trueDeriv, deriv)
	}
}
