package par

import (
	"testing"

	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/nnet"
	"github.com/gonum/floats"
	"math/rand"

	//"fmt"
)

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

func DoParAndSeqMatch(inputs, truths [][]float64, net *nnet.Net, chunkSize int) bool {
	net.RandomizeParameters()
	dLossDParamSeq, dLossDParamSeqFlat := net.NewPerParameterMemory()
	p := NewParLossDerivMemory(net)
	SeqLossDeriv(inputs, truths, net, dLossDParamSeq, p)
	dLossDParamPar, dLossDParamParFlat := net.NewPerParameterMemory()
	ParLossDeriv(inputs, truths, net, dLossDParamPar, chunkSize)

	floats.Scale(float64(1/len(inputs)), dLossDParamParFlat)
	floats.Scale(float64(1/len(inputs)), dLossDParamSeqFlat)

	return floats.Eq(dLossDParamParFlat, dLossDParamSeqFlat, 1e-14)
}

func TestParLossDeriv(t *testing.T) {
	net := nnet.DefaultRegression(3, 2, 2, 4)
	net.Losser = loss.SquaredDistance{}
	nInputs := 1000
	inputs := RandomSliceOfSlice(nInputs, 3)
	truths := RandomSliceOfSlice(nInputs, 2)
	chunkSize := 8
	if !DoParAndSeqMatch(inputs, truths, net, chunkSize) {
		t.Errorf("Par and seq don't match")
	}
}
