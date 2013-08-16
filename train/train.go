package train

import (
	"github.com/btracey/gofunopter/common"
	"github.com/btracey/gofunopter/common/display"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/nnet"
	"github.com/btracey/nnet/scale"

	"github.com/gonum/floats"
	"math/rand"
	"sync"

	"fmt"
	"runtime"
)

var TestLossIncrease common.Status = 100

// GetChunkSize returns the number of inputs per parallel goroutine
func GetChunkSize(inputs int) int {
	nCPU := runtime.NumCPU()

	chunkSize := inputs / nCPU
	if chunkSize < 5 {
		chunkSize = 5
	}
	if chunkSize > 1000 {
		chunkSize = 1000
	}
	return chunkSize
}

func SetScale(inputs, outputs [][]float64, net *nnet.Net) {
	net.InputMean, net.InputStd = scale.FindScale(inputs)
	net.OutputMean, net.OutputStd = scale.FindScale(outputs)
}

// OneFoldTrain trains the net by splitting the data into a training and a testing
// fold and stopping when the testing fold has
type OneFoldTrain struct {
	LossRatio float64

	net          *nnet.Net
	trainInputs  [][]float64
	trainOutputs [][]float64
	testInputs   [][]float64
	testOutputs  [][]float64
	chunkSize    int

	dLossDParamTrain     [][][]float64
	dLossDParamTrainFlat []float64
	dLossDParamTest      [][][]float64
	dLossDParamTestFlat  []float64
	trainLoss            float64
	testLoss             float64

	lossRatio float64
}

func NewOneFoldTrain(net *nnet.Net, losser loss.Losser, inputs, outputs [][]float64) *OneFoldTrain {

	if len(inputs) != len(outputs) {
		panic("input and output lengths must match")
	}

	net.Losser = losser
	chunkSize := GetChunkSize(len(inputs))

	o := &OneFoldTrain{
		net:       net,
		chunkSize: chunkSize,
		LossRatio: 7,
	}

	// Divide the inputs and outputs in half
	nSamples := len(inputs)
	nInputs := len(inputs[0])
	nOutputs := len(outputs[0])
	nTest := int(float64(nSamples) / 2)
	nTrain := nSamples - nTest
	rp := rand.Perm(nSamples)
	o.testInputs = make([][]float64, nTest)
	o.testOutputs = make([][]float64, nTest)
	o.trainInputs = make([][]float64, nTrain)
	o.trainOutputs = make([][]float64, nTrain)

	o.dLossDParamTrain, o.dLossDParamTrainFlat = net.NewPerParameterMemory()
	o.dLossDParamTest, o.dLossDParamTestFlat = net.NewPerParameterMemory()

	var n int
	// Copy the test inputs over
	for i := range o.testInputs {
		o.testInputs[i] = make([]float64, nInputs)
		n = copy(o.testInputs[i], inputs[rp[i]])
		if n != nInputs {
			panic("all inputs must have the same length")
		}
		o.testOutputs[i] = make([]float64, nOutputs)
		n = copy(o.testOutputs[i], outputs[rp[i]])
		if n != nOutputs {
			panic("all outputs must have the same length")
		}
	}
	// Copy the train inputs over
	for i := range o.trainInputs {
		o.trainInputs[i] = make([]float64, nInputs)
		n = copy(o.trainInputs[i], inputs[rp[nTest+i]])
		if n != nInputs {
			panic("all inputs must have the same length")
		}
		o.trainOutputs[i] = make([]float64, nOutputs)
		n = copy(o.trainOutputs[i], outputs[rp[nTest+i]])
		if n != nOutputs {
			panic("all outputs must have the same length")
		}
	}

	//fmt.Println("inputs", inputs)
	//fmt.Println("trainInputs", o.trainInputs)
	//fmt.Println("testInputs", o.testInputs)
	return o
}

func (o *OneFoldTrain) Init() error {

	fmt.Println("One fold train initialize")
	SetScale(o.trainInputs, o.trainOutputs, o.net)
	scale.ScaleData(o.trainInputs, o.net.InputMean, o.net.InputStd)
	scale.ScaleData(o.trainOutputs, o.net.OutputMean, o.net.OutputStd)
	scale.ScaleData(o.testInputs, o.net.InputMean, o.net.InputStd)
	scale.ScaleData(o.testOutputs, o.net.OutputMean, o.net.OutputStd)
	fmt.Println("Done scale")

	/*
		fmt.Println(o.net.InputMean)
		fmt.Println(o.net.InputStd)
		fmt.Println(o.net.OutputMean)
		fmt.Println(o.net.OutputStd)
	*/

	/*
		for i := range o.trainInputs {
			fmt.Println(o.trainInputs[i])
		}
		for i := range o.trainOutputs {
			fmt.Println(o.trainOutputs[i])
		}
	*/
	if o.net.Losser == nil {
		return fmt.Errorf("Net does not have Losser set")
	}
	return nil
}

func (o *OneFoldTrain) Result() {
	scale.UnscaleData(o.trainInputs, o.net.InputMean, o.net.InputStd)
	scale.UnscaleData(o.testInputs, o.net.InputMean, o.net.InputStd)
	scale.UnscaleData(o.trainOutputs, o.net.OutputMean, o.net.OutputStd)
	scale.UnscaleData(o.testOutputs, o.net.OutputMean, o.net.OutputStd)
}

func (o *OneFoldTrain) Status() common.Status {
	if o.lossRatio > o.LossRatio {
		return TestLossIncrease
	}
	return common.Continue
}

func (o *OneFoldTrain) ObjGrad(weights []float64) (loss float64, deriv []float64, err error) {

	o.net.SetParametersSlice(weights)

	trainLossChan := make(chan float64)
	testLossChan := make(chan float64)
	go func() {
		trainLossChan <- nnet.ParLossDeriv(o.trainInputs, o.trainOutputs, o.net, o.dLossDParamTrain, o.chunkSize)
	}()
	go func() {
		testLossChan <- nnet.ParLossDeriv(o.testInputs, o.testOutputs, o.net, o.dLossDParamTest, o.chunkSize)
	}()

	w := sync.WaitGroup{}
	w.Add(2)
	go func() {
		o.trainLoss = <-trainLossChan
		o.trainLoss /= float64(len(o.trainInputs))
		//fmt.Println("Train 2 norm", floats.Norm(dLossDParamTrainFlat, 2))
		floats.Scale(1/float64(len(o.trainInputs)), o.dLossDParamTrainFlat)
		//fmt.Println("Train 2 norm", floats.Norm(dLossDParamTrainFlat, 2))
		w.Done()
	}()
	go func() {
		o.testLoss = <-testLossChan
		o.testLoss /= float64(len(o.testInputs))

		floats.Scale(1/float64(len(o.testInputs)), o.dLossDParamTestFlat)
		w.Done()
	}()
	w.Wait()

	o.lossRatio = o.testLoss / o.trainLoss

	//fmt.Println("Done waiting")
	//fmt.Println("trainLoss", o.trainLoss)
	//fmt.Println("nTrain", float64(len(o.trainInputs)))
	//fmt.Println("nTest", float64(len(o.testInputs)))
	return o.trainLoss, o.dLossDParamTrainFlat, nil
}

func (o *OneFoldTrain) AddToDisplay(d []*display.Struct) []*display.Struct {
	return append(d, &display.Struct{Heading: "TestLoss", Value: o.testLoss}, &display.Struct{Heading: "LossRatio", Value: o.lossRatio})
}
