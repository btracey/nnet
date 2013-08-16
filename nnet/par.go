// package par implements computing loss and derivative in parallel. Right now just has shared-memory parallel,
// but should eventually have more
package nnet

import (
//"fmt"
)

type ParLossDerivMemory struct {
	derivTmp           *PredLossDerivTmpMemory
	predictionTmp      []float64
	dLossDParamTmp     [][][]float64
	dLossDParamTmpFlat []float64
}

func NewParLossDerivMemory(net *Net) *ParLossDerivMemory {
	p := &ParLossDerivMemory{
		derivTmp:      net.NewPredLossDerivTmpMemory(),
		predictionTmp: make([]float64, net.Outputs()),
	}
	p.dLossDParamTmp, p.dLossDParamTmpFlat = net.NewPerParameterMemory()
	return p
}

func SeqLossDeriv(inputs, truths [][]float64, net *Net, dLossDParam [][][]float64, p *ParLossDerivMemory) (loss float64) {
	// Compute first loss and store in it dLossDParam
	loss = PredLossDeriv(inputs[0], truths[0], net, p.derivTmp, p.predictionTmp, dLossDParam)
	// Sum up the next losses and derivatives
	for i := 1; i < len(inputs); i++ {
		newLoss := PredLossDeriv(inputs[i], truths[i], net, p.derivTmp, p.predictionTmp, p.dLossDParamTmp)

		//fmt.Println("input", inputs[i])
		//fmt.Println("truth", truths[i])
		//fmt.Println("pred", p.predictionTmp)
		loss += newLoss
		for i, lay := range p.dLossDParamTmp {
			for j, neur := range lay {
				for k, val := range neur {
					dLossDParam[i][j][k] += val
				}
			}
		}
	}
	return loss
}

type Result struct {
	dLossDParam [][][]float64
	loss        float64
}

func ParLossDeriv(inputs, truths [][]float64, net *Net, dLossDParam [][][]float64, chunkSize int) (loss float64) {
	// Zero out dLossDParam
	for i, lay := range dLossDParam {
		for j, neur := range lay {
			for k := range neur {
				dLossDParam[i][j][k] = 0
			}
		}
	}

	receiveChan := make(chan *Result, 10) // Add a buffer so there is no blocking

	count := 0
	nSent := 0
	var startInd, endInd int
	for {
		startInd = count
		if count+chunkSize > len(inputs) {
			endInd = len(inputs)
		} else {
			endInd = count + chunkSize
		}
		nSent++
		go func(startInd, endInd int, c chan *Result) {
			dLossDParam, _ := net.NewPerParameterMemory()
			loss := SeqLossDeriv(inputs[startInd:endInd], truths[startInd:endInd], net, dLossDParam, NewParLossDerivMemory(net))
			c <- &Result{loss: loss, dLossDParam: dLossDParam}
		}(startInd, endInd, receiveChan)

		if endInd == len(inputs) {
			break
		}
		count += chunkSize
	}

	for i := 0; i < nSent; i++ {
		r := <-receiveChan
		loss += r.loss
		for i, lay := range r.dLossDParam {
			for j, neur := range lay {
				for k, val := range neur {
					dLossDParam[i][j][k] += val
				}
			}
		}
	}
	return
}
