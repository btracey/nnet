package nnet

import (
	"testing"

	"github.com/btracey/nnet/loss"
	"github.com/gonum/floats"
	"math/rand"

	//"fmt"
)

func RandomWeights(first int) []float64 {
	weights := make([]float64, first)
	for i := range weights {
		weights[i] = 20 * rand.Float64()
	}
	return weights
}

func RandomSliceOfSlice(first, second int) (sos [][]float64) {
	sos = make([][]float64, first)
	for i := range sos {
		sos[i] = make([]float64, second)
		for j := range sos[i] {
			sos[i][j] = rand.NormFloat64()
		}
	}
	return
}

func DoParAndSeqMatch(inputs, truths [][]float64, weights []float64, net *Net, chunkSize int) bool {
	net.RandomizeParameters()
	dLossDParamSeq, dLossDParamSeqFlat := net.NewPerParameterMemory()
	p := NewParLossDerivMemory(net)
	SeqLossDeriv(inputs, truths, weights, net, dLossDParamSeq, p)
	dLossDParamPar, dLossDParamParFlat := net.NewPerParameterMemory()
	ParLossDeriv(inputs, truths, weights, net, dLossDParamPar, chunkSize)

	floats.Scale(float64(1/len(inputs)), dLossDParamParFlat)
	floats.Scale(float64(1/len(inputs)), dLossDParamSeqFlat)

	return floats.EqualApprox(dLossDParamParFlat, dLossDParamSeqFlat, 1e-14)
}

func TestParLossDeriv(t *testing.T) {
	net := DefaultRegression(3, 2, 2, 4)
	net.Losser = loss.SquaredDistance{}
	nInputs := 1000
	inputs := RandomSliceOfSlice(nInputs, 3)
	truths := RandomSliceOfSlice(nInputs, 2)
	weights := RandomWeights(nInputs)
	chunkSize := 8
	if !DoParAndSeqMatch(inputs, truths, weights, net, chunkSize) {
		t.Errorf("Par and seq don't match")
	}
}
