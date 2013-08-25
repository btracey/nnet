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
	net.InputScaler.SetScale(inputs)
	net.OutputScaler.SetScale(outputs)
}

// OneFoldTrain trains the net by splitting the data into a training and a testing
// fold and stopping when the testing fold has
type OneFoldTrain struct {
	LossRatio float64

	net          *nnet.Net
	TrainInputs  [][]float64
	TrainOutputs [][]float64
	TestInputs   [][]float64
	TestOutputs  [][]float64
	chunkSize    int

	dLossDParamTrain     [][][]float64
	dLossDParamTrainFlat []float64
	dLossDParamTest      [][][]float64
	dLossDParamTestFlat  []float64
	trainLoss            float64
	testLoss             float64

	lossRatio float64

	KeepHistory bool // Keep the test loss history
	History     []float64
	nCalls      int
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
	o.TestInputs = make([][]float64, nTest)
	o.TestOutputs = make([][]float64, nTest)
	o.TrainInputs = make([][]float64, nTrain)
	o.TrainOutputs = make([][]float64, nTrain)

	o.dLossDParamTrain, o.dLossDParamTrainFlat = net.NewPerParameterMemory()
	o.dLossDParamTest, o.dLossDParamTestFlat = net.NewPerParameterMemory()

	var n int
	// Copy the test inputs over
	for i := range o.TestInputs {
		o.TestInputs[i] = make([]float64, nInputs)
		n = copy(o.TestInputs[i], inputs[rp[i]])
		if n != nInputs {
			panic("all inputs must have the same length")
		}
		o.TestOutputs[i] = make([]float64, nOutputs)
		n = copy(o.TestOutputs[i], outputs[rp[i]])
		if n != nOutputs {
			panic("all outputs must have the same length")
		}
	}
	// Copy the train inputs over
	for i := range o.TrainInputs {
		o.TrainInputs[i] = make([]float64, nInputs)
		n = copy(o.TrainInputs[i], inputs[rp[nTest+i]])
		if n != nInputs {
			panic("all inputs must have the same length")
		}
		o.TrainOutputs[i] = make([]float64, nOutputs)
		n = copy(o.TrainOutputs[i], outputs[rp[nTest+i]])
		if n != nOutputs {
			panic("all outputs must have the same length")
		}
	}

	//fmt.Println("inputs", inputs)
	//fmt.Println("TrainInputs", o.TrainInputs)
	//fmt.Println("TestInputs", o.TestInputs)
	return o
}

func (o *OneFoldTrain) Init() error {

	fmt.Println("One fold train initialize")
	SetScale(o.TrainInputs, o.TrainOutputs, o.net)
	scale.ScaleData(o.net.InputScaler, o.TrainInputs)
	scale.ScaleData(o.net.OutputScaler, o.TrainOutputs)
	scale.ScaleData(o.net.InputScaler, o.TestInputs)
	scale.ScaleData(o.net.OutputScaler, o.TestOutputs)
	fmt.Println("Done scale")

	/*
		fmt.Println(o.net.InputMean)
		fmt.Println(o.net.InputStd)
		fmt.Println(o.net.OutputMean)
		fmt.Println(o.net.OutputStd)
	*/

	/*
		for i := range o.TrainInputs {
			fmt.Println(o.TrainInputs[i])
		}
		for i := range o.TrainOutputs {
			fmt.Println(o.TrainOutputs[i])
		}
	*/
	if o.net.Losser == nil {
		return fmt.Errorf("Net does not have Losser set")
	}
	return nil
}

func (o *OneFoldTrain) Result() {
	scale.UnscaleData(o.net.InputScaler, o.TrainInputs)
	scale.UnscaleData(o.net.InputScaler, o.TestInputs)
	scale.UnscaleData(o.net.OutputScaler, o.TrainOutputs)
	scale.UnscaleData(o.net.OutputScaler, o.TestOutputs)
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
		trainLossChan <- nnet.ParLossDeriv(o.TrainInputs, o.TrainOutputs, o.net, o.dLossDParamTrain, o.chunkSize)
	}()
	go func() {
		testLossChan <- nnet.ParLossDeriv(o.TestInputs, o.TestOutputs, o.net, o.dLossDParamTest, o.chunkSize)
	}()

	w := sync.WaitGroup{}
	w.Add(2)
	go func() {
		o.trainLoss = <-trainLossChan
		o.trainLoss /= float64(len(o.TrainInputs))
		//fmt.Println("Train 2 norm", floats.Norm(dLossDParamTrainFlat, 2))
		floats.Scale(1/float64(len(o.TrainInputs)), o.dLossDParamTrainFlat)
		//fmt.Println("Train 2 norm", floats.Norm(dLossDParamTrainFlat, 2))
		w.Done()
	}()
	go func() {
		o.testLoss = <-testLossChan
		o.testLoss /= float64(len(o.TestInputs))

		floats.Scale(1/float64(len(o.TestInputs)), o.dLossDParamTestFlat)
		w.Done()
	}()
	w.Wait()

	o.lossRatio = o.testLoss / o.trainLoss

	o.AddToHistory(o.testLoss)
	//	fmt.Println()

	//fmt.Println("Done waiting")
	//fmt.Println("trainLoss", o.trainLoss)
	//fmt.Println("nTrain", float64(len(o.TrainInputs)))
	//fmt.Println("nTest", float64(len(o.TestInputs)))
	return o.trainLoss, o.dLossDParamTrainFlat, nil
}

func (o *OneFoldTrain) AddToDisplay(d []*display.Struct) []*display.Struct {
	return append(d, &display.Struct{Heading: "TestLoss", Value: o.testLoss}, &display.Struct{Heading: "LossRatio", Value: o.lossRatio})
}

func (o *OneFoldTrain) AddToHistory(testLoss float64) {
	if o.KeepHistory {
		o.History = append(o.History, testLoss)
	}
}
