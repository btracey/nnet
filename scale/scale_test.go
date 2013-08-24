package scale

import (
	"github.com/gonum/floats"
	"testing"

	"fmt"
)

func testLinear(t *testing.T, kind linearTest) {
	u := &Linear{}
	err := u.SetScale(kind.data)
	fmt.Println("Error test nil?", err == nil)

	if err != nil {
		if kind.eqDim != true {
			t.Errorf("Error where there shouldn't be for case " + kind.name + ": " + err.Error())
		}
	}
	if !floats.EqualApprox(u.Min, kind.min, 1e-14) {
		t.Errorf("Min doesn't match for case " + kind.name)
	}
	if !floats.EqualApprox(u.Max, kind.max, 1e-14) {
		t.Errorf("Max doesn't match for case " + kind.name)
	}
}

type linearTest struct {
	data  [][]float64
	min   []float64
	max   []float64
	name  string
	eqDim bool
}

func TestLinear(t *testing.T) {
	cases := []linearTest{
		linearTest{
			data: [][]float64{
				[]float64{1},
				[]float64{2},
				[]float64{-3},
				[]float64{-4},
			},
			min:  []float64{-4},
			max:  []float64{2},
			name: "OneD",
		},
		linearTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 9},
				[]float64{-3, 12},
				[]float64{-4, 15},
			},
			min:  []float64{-4, 4},
			max:  []float64{2, 15},
			name: "TwoD",
		},
		linearTest{
			data: [][]float64{
				[]float64{1, 4},
				[]float64{2, 4},
				[]float64{-3, 4},
				[]float64{-4, 4},
			},
			min:   []float64{-4, 3.5},
			max:   []float64{2, 4.5},
			name:  "EqDim",
			eqDim: true,
		},
	}
	for i := range cases {
		testLinear(t, cases[i])
	}
}
